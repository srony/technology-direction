"""Rate a handful of triplets from the grid and print the answers. Two triplets per domain, two constructs, three
personas, one draw: 24 calls to the open-weight model.

    set OPENROUTER_API_KEY=...
    python demo.py
"""
import pandas as pd

import llm
from prompts import PERSONAS, STATEMENTS, system_prompt, user_prompt

MODEL, T, P = "deepseek/deepseek-chat-v3.1", 0.3, 0.3

grid = pd.read_csv("grid.csv", dtype={"isco3": str, "section": str})
pick = pd.concat([grid[grid.domain == d].sample(2, random_state=7) for d in ("digital", "green")])
hdrs = llm.headers("direction panel demo")
for _, row in pick.iterrows():
    print(f"\n{row.domain}: {row.node_label} | {row.occ_title} | {row.sec_title}")
    for construct in STATEMENTS:
        for persona in PERSONAS:
            res = llm.ask(user_prompt(row, construct, persona), system_prompt(row.domain), MODEL, T, P, hdrs)
            rating = None if res is None else res.get("rating")
            reason = "" if res is None else res.get("reason", "")
            print(f"  {construct[:4]} {persona['name']:16s} {rating}  {reason}")
