"""Build the knowledge base (data/tickets.json) from the synthetic Excel dataset.

Every ticket description shares the same ~28 lines of generic troubleshooting
boilerplate; only the first line describes the actual issue. We keep that line
as the ticket summary so retrieval and the LLM focus on what is unique.

Usage:  python scripts/build_kb.py
"""
import json
from collections import Counter
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "synthetic_data.xlsx"
OUT = ROOT / "data" / "tickets.json"


def main():
    ws = openpyxl.load_workbook(SRC, read_only=True, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    body = rows[1:]

    tails = Counter("\n".join(r[2].split("\n")[1:]) for r in body)
    (boilerplate, _), = tails.most_common(1)

    tickets = []
    for ticket_id, subject, description, department, reported in body:
        lines = description.split("\n")
        tickets.append({
            "id": ticket_id,
            "subject": subject.strip(),
            "summary": lines[0].strip(),
            "department": department.strip(),
            "date": reported.strftime("%Y-%m-%d"),
        })

    OUT.write_text(json.dumps({
        "tickets": tickets,
        "standard_procedure": boilerplate.split("\n"),
    }, indent=2))
    print(f"Wrote {len(tickets)} tickets to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
