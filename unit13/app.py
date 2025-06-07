"""Run the Bitcoin Q&A agent via CLI."""

from __future__ import annotations

import sys

from agents.workflows.qa_workflow import default_agent, run_query


def main() -> None:
    agent = default_agent()
    print("Bitcoin Q&A Agent. Type a question or 'exit'.")
    for line in sys.stdin:
        question = line.strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        print(f"\033[91m{question}\033[0m")
        print(f"\033[92m{run_query(agent, question)}\033[0m")


if __name__ == "__main__":
    main()
