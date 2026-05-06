"""
VEDA — Autonomous Data Science System
main.py — Entry point with self-healing health monitor
"""

from veda.core.graph import run_veda
from veda.monitors.health_monitor import AgentHealthMonitor
from veda.idle_pool.idle_agents import build_idle_pool

if __name__ == "__main__":

    GOAL = "predict whether a customer review is positive or negative. target: sentiment"
    DATASET = "data/reviews_10m.csv"

    # ── Build idle pool ───────────────────────────────────────
    print("\n[VEDA] Building idle agent pool...")
    idle_pool = build_idle_pool()
    print("[VEDA] Idle pool ready — " + str(len(idle_pool)) + " agents on standby")

    # ── Start health monitor ──────────────────────────────────
    monitor = AgentHealthMonitor(check_interval=30, critical_threshold=35.0)

    for domain, agent in idle_pool.items():
        monitor.register_idle(agent, domain)

    monitor.start()
    print("[VEDA] Health monitor started\n")

    # ── Run pipeline ──────────────────────────────────────────
    result = run_veda(goal=GOAL, dataset_path=DATASET)

    # ── Print health status ───────────────────────────────────
    monitor.print_status()

    # ── Stop monitor ──────────────────────────────────────────
    monitor.stop()

    # ── Print results ─────────────────────────────────────────
    print("\n── Execution Plan ──")
    plan = result.get("execution_plan", {})
    if isinstance(plan, dict):
        print("Task type : " + str(plan.get("task_type", "unknown")))
        steps = plan.get("steps", [])
        print("Steps     : " + str([s.get("agent_name", s) if isinstance(s, dict) else s for s in steps]))

    print("\n── Model Results ──")
    model_info = result.get("model_info", {})
    metrics = model_info.get("test_metrics", {})
    print("Model    : " + str(model_info.get("model_name", "N/A")))
    print("AUC-ROC  : " + str(metrics.get("auc_roc", "N/A")))
    print("F1 Score : " + str(metrics.get("f1_score", "N/A")))
    print("Accuracy : " + str(metrics.get("accuracy", "N/A")))

    print("\n── Outputs ──")
    outputs = result.get("outputs", {})
    print("Dashboard : " + str(outputs.get("dashboard_path", "N/A")))
    print("Report    : " + str(outputs.get("executive_report_path", "N/A")))