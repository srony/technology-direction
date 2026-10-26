"""The prompts of the direction panel, verbatim. One construct and one persona per call."""

PERSONAS = [
    {"name": "working_manager", "role": "a manager who runs the team that performs this work day to day"},
    {"name": "worker", "role": "an experienced worker who personally performs this task day to day"},
    {"name": "ceo", "role": "a chief executive officer responsible for the organisation's strategy and investment decisions"},
]

SYSTEM = (
    "You are an expert in {domain} technologies. Your expertise is defined by the given persona and context in user prompt.\n"
    "\n"
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
    "- 0 means you do not know. Never use 0 to signal a refusal."
)

PREAMBLE = """You are assessing how a {d} technology will affect an occupation as it is practised within a specific industry.
- Substitute: the {d} technology takes over the work itself, so fewer human workers are needed to do it.
- Complement: the {d} technology supports or enhances the human workers, who still do the work, but better, faster or at higher quality.

"""
BACKGROUND = """You are {role}. Answer strictly from that perspective.

"""
STATEMENTS = {
    "substitutability": "This {d} technology will replace human workers in this occupation within this industry.",
    "complementarity": "This {d} technology will work alongside human workers in this occupation within this industry and support them.",
}
SCALE = """To what extent do you agree with the statement below?
- 1 = Strongly disagree
- 2 = Disagree (25 percent)
- 3 = Neutral / somewhat agree (50 percent)
- 4 = Mostly agree (75 percent)
- 5 = Fully agree
- 0 = Do not know

Statement: {statement}

Provide exactly one rating from [0, 1, 2, 3, 4, 5] and one concise reason (1-3 sentences).
"""


def system_prompt(domain):
    return SYSTEM.format(domain=domain)


def user_prompt(row, construct, persona):
    """row: a grid row with domain, occ_title, sec_title, node_label."""
    d = row["domain"]
    ctx = (f"Occupation: {row['occ_title']}\n"
           f"Industry: {row['sec_title']}\n"
           f"{d.capitalize()} technology: {row['node_label']}\n\n")
    return PREAMBLE.format(d=d) + BACKGROUND.format(**persona) + ctx + SCALE.format(statement=STATEMENTS[construct].format(d=d))
