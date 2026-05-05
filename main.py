"""
VEDA — Autonomous Data Science System
main.py — Entry point

Run VEDA with:
    python main.py
"""

from veda.core.graph import run_veda

if __name__ == "__main__":

    # ── Test run configuration ───────────────────────────────
    GOAL = "predict whether a passenger survived the Titanic. target: Survived"
    DATASET = "data/titanic.csv"

    # ── Run VEDA ─────────────────────────────────────────────
    result = run_veda(goal=GOAL, dataset_path=DATASET)

    # ── Print results ─────────────────────────────────────────
    print("\n── Execution Plan ──")
    plan = result.get("execution_plan", {})
    print(f"Task type : {plan.get('task_type', 'unknown')}")
    print(f"Steps     : {[s['agent_name'] for s in plan.get('steps', [])]}")

    print("\n── Data Profile ──")
    profile = result.get("data_profile", {})
    print(f"Rows      : {profile.get('row_count', 0)}")
    print(f"Columns   : {profile.get('col_count', 0)}")
    print(f"Target    : {profile.get('target_column', 'none')}")

    print("\n── Decision Log ──")
    for log in result.get("planner_decision_log", []):
        print(f"  {log}")