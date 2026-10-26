"""
shared/llm.py
~~~~~~~~~~~~~
Thin wrappers around the OpenRouter chat-completion API.

Every path operation uses ``pathlib.Path`` so the code is portable
across Windows and macOS.
"""

from __future__ import annotations

import json
import re
import time

import requests


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------

def _parse_choice_message(choice_msg):
    """Return {'rating': int|None, 'reason': str} from OpenRouter choice.message."""
    # Prefer parsed (when using response_format: json_schema)
    if isinstance(choice_msg.get("parsed"), dict):
        out = choice_msg["parsed"]
        return {"rating": out.get("rating"), "reason": out.get("reason", "")}

    # Fallback: JSON string in content
    content = (choice_msg.get("content") or "").strip()
    if not content:
        return {"rating": None, "reason": ""}

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"rating": None, "reason": content}


# ------------------------------------------------------------------
# OpenRouter caller
# ------------------------------------------------------------------

DEFAULT_SYSTEM = (
    "You are an expert in digital technologies. Your expertise is defined by the given persona and context in user prompt.\n"
    "\n You are responding to Delphi Survey in the year 2023 \n\n"
    "Task: Given a prompt, produce exactly ONE JSON object evaluating the statement.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "- Output MUST be a single JSON object with exactly these keys in this order:\n"
    "  1) \"rating\": an integer from 0 to 5 (no decimals).\n"
    "  2) \"reason\": a 1-3 lines string (<= 25 words), plain text.\n"
    "- No additional keys, no trailing commas.\n"
    "- No prose before or after the JSON. No code fences. No comments. No markdown.\n\n"
    "POLICY:\n"
    "- If information is ambiguous, choose the closest rating and explain briefly.\n"
    "- Never ask questions or request more info.\n"
    "- Use a neutral, expert tone; avoid hedging unless required by the persona.\n"
    "- If you cannot comply, still return a JSON object with \"rating\": 0 and a \"reason\" describing the constraint."
)


def ask_openrouter(
    prompt,
    model="openai/gpt-4o-mini",
    *,
    temperature=1,
    top_p=1,
    max_tokens=200,
    max_retries=2,
    timeout_s=180,
    system_prompt=None,
    HEADERS,
    OPENROUTER_URL="https://openrouter.ai/api/v1/chat/completions"
):
    """
    Call OpenRouter with schema-enforced JSON. On failures/timeout returns None.

    Args:
        prompt (str): The survey prompt to send.
        model (str): Model name (default = "openai/gpt-4o-mini").
        temperature (float): Sampling temperature (0 = deterministic).
        top_p (float): Nucleus sampling cutoff (0-1).
        max_tokens (int): Maximum tokens to generate.
        max_retries (int): How many times to retry on failure.
        timeout_s (int): Request timeout in seconds.
    """
    payload = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "SurveyAnswer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "rating": {"type": "integer", "minimum": 0, "maximum": 5},
                        "reason": {"type": "string"}
                    },
                    "required": ["rating", "reason"],
                    "additionalProperties": False
                }
            }
        }
    }

    for attempt in range(1, max_retries + 2):  # attempts 1..(max_retries+1)
        try:
            r = requests.post(
                OPENROUTER_URL, headers=HEADERS, json=payload, timeout=timeout_s
            )
            if not r.ok:
                return None
            resp = r.json()
            choice_msg = resp["choices"][0]["message"]
            return _parse_choice_message(choice_msg)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            if attempt <= max_retries:
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            return None


# ------------------------------------------------------------------
# Generic OpenRouter chat (used by P2 taxonomy labeling)
# ------------------------------------------------------------------

def openrouter_chat(
    api_key: str,
    model: str,
    user_agent: str,
    prompt: str,
    temperature: float = 0.01,
    max_tokens: int = 64,
    timeout_s: int = 60,
    system_prompt: str | None = None,
    json_schema: dict | None = None,
) -> str:
    """OpenRouter chat call that returns the assistant text.

    When *json_schema* is provided the API enforces structured output via
    ``response_format: {type: "json_schema", ...}``.  The schema dict must
    follow the OpenAI-style JSON-schema spec (``name``, ``strict``,
    ``schema`` keys).

    Parameters
    ----------
    system_prompt : str or None
        Custom system prompt.  When ``None`` the default taxonomy-labelling
        prompt is used.
    json_schema : dict or None
        If provided, enforce structured JSON output via the API.
    """
    if system_prompt is None:
        system_prompt = (
            "You label clusters for a technology taxonomy. "
            "Return ONE short, specific label (1-4 words). No quotes. "
            "Prefer concrete subfield names over generic terms (except in root)."
        )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": user_agent,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    if json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"].strip()
    if json_schema is None:
        # Only strip quotes/whitespace for free-text responses
        text = re.sub(r'^[\'"\s]+|[\'"\s]+$', "", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------------
# API key loading
# ------------------------------------------------------------------

def read_openrouter_key() -> str:
    """Read the OpenRouter API key from the standard Dropbox location."""
    from pathlib import Path

    home = Path.home()
    key_path = home / "Dropbox" / "Work" / "00_keys" / "openrouter_key.txt"
    if not key_path.exists():
        raise FileNotFoundError(f"OpenRouter key not found at: {key_path}")
    return key_path.read_text(encoding="utf-8").strip()
