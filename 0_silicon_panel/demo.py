"""Rate four illustrative cases with the silicon panel and print the answers. Four personas, four statements each, one
draw: 16 calls to the open-weight model at the knee configuration. The personas and cases are invented for the
demonstration and are not records of the survey.

    set OPENROUTER_API_KEY=...
    python demo.py
"""
import llm
from prompts import STATEMENTS, SYSTEM_PROMPT, user_prompt

MODEL, T, P = "deepseek/deepseek-chat-v3.1", 0.3, 0.3

CASES = [
    dict(gender="Female", role="Researcher", organization="University", educbackground="Computer science",
         education="Doctorate", region="Western Europe", tech="Collaborative robots",
         industry="Manufacture of machinery and equipment", task="Assemble machine components"),
    dict(gender="Male", role="Developer of automation technologies", organization="Private company",
         educbackground="Engineering", education="Master", region="Northern Europe", tech="Large language models",
         industry="Legal and accounting activities", task="Draft standard contracts"),
    dict(gender="Male", role="User of automation technologies", organization="Public sector", educbackground="Economics",
         education="Master", region="Southern Europe", tech="Computer vision", industry="Crop and animal production",
         task="Inspect crops for disease"),
    dict(gender="Female", role="Policy maker", organization="Government", educbackground="Social sciences",
         education="Doctorate", region="Eastern Europe", tech="Robotic process automation",
         industry="Public administration", task="Process benefit applications"),
]

hdrs = llm.headers("survey panel demo")
for case in CASES:
    print(f"\n{case['tech']} | {case['industry']} | {case['task']}  ({case['role']}, {case['region']})")
    for q in STATEMENTS:
        res = llm.ask(user_prompt({**case, "question": q}), SYSTEM_PROMPT, MODEL, T, P, hdrs)
        rating = None if res is None else res.get("rating")
        reason = "" if res is None else res.get("reason", "")
        print(f"  {q:16s} {rating}  {reason}")
