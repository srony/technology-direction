"""OpenRouter chat call with schema-enforced JSON. The key comes from the environment variable OPENROUTER_API_KEY."""
import json
import os
import re
import time

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "SurveyAnswer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"rating": {"type": "integer", "minimum": 0, "maximum": 5}, "reason": {"type": "string"}},
            "required": ["rating", "reason"],
            "additionalProperties": False,
        },
    },
}


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY in the environment")
    return key


def headers(title="direction panel"):
    return {"Authorization": f"Bearer {api_key()}", "HTTP-Referer": "http://localhost", "X-Title": title,
            "Content-Type": "application/json"}


def _parse(choice_msg):
    if isinstance(choice_msg.get("parsed"), dict):
        out = choice_msg["parsed"]
        return {"rating": out.get("rating"), "reason": out.get("reason", "")}
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


def ask(prompt, system, model, temperature, top_p, hdrs, max_tokens=200, max_retries=2, timeout_s=180):
    """One rating call. Returns {'rating': int|None, 'reason': str} or None on failure."""
    payload = {"model": model, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
               "response_format": SCHEMA}
    for attempt in range(1, max_retries + 2):
        try:
            r = requests.post(URL, headers=hdrs, json=payload, timeout=timeout_s)
            if not r.ok:
                return None
            return _parse(r.json()["choices"][0]["message"])
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            if attempt <= max_retries:
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            return None
