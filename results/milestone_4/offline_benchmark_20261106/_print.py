import json
from pathlib import Path

r = json.loads(Path("benchmark.json").read_text(encoding="utf-8"))
print("live_jev", r["live_jev_calls"], "iters", r["bootstrap_iterations"])


def show(section: str) -> None:
    for name, s in r[section].items():
        print(
            section,
            name,
            "mean%",
            round(s["mean_saving_percent"], 4),
            "med%",
            round(s["median_saving_percent"], 4),
            "worse",
            s["workloads_worse"],
            "keep",
            s["workloads_retaining_original"],
            "acc",
            s["selection_accuracy"],
            "regret",
            s["mean_regret_usd"],
            "maxr",
            s["maximum_regret_usd"],
            "prof",
            s["total_profiling_cost_usd"],
            "abst",
            s["abstention_or_fallback_count"],
        )
        for wid, w in s["workloads"].items():
            print(
                "   ",
                wid,
                w.get("decision"),
                "save%",
                w.get("observed_saving_percent"),
                "cheapest",
                w.get("selected_is_observed_cheapest"),
                "order",
                " -> ".join(w["selected_order"]),
            )


show("strategies")
print()
show("ablations")
print()
for w in r["workloads"]:
    sel = w["selections"]["risk_aware"]
    expl = sel.get("explanation") or {}
    print(
        "RISK",
        w["workload_id"],
        sel["decision"],
        sel.get("reason"),
        "P",
        expl.get("probability_cheaper_than_original"),
        "rel",
        expl.get("expected_relative_saving"),
        "cvar",
        expl.get("cvar_loss_usd"),
        "regret",
        expl.get("expected_regret_usd"),
    )
    rob = w["selections"]["risk_aware_robust"]
    print("ROB", w["workload_id"], rob["decision"], rob.get("reason"))
    print("ADAPT", w["workload_id"], w["selections"]["risk_aware_adaptive"]["decision"], w["adaptive"])
    print("ECON", w["workload_id"], w["selections"]["risk_aware_economics"]["decision"])
    print(
        "var",
        w["response_variability"]["status"],
        w["response_variability"].get("total_recorded_threshold_crossings"),
    )
    print(
        "hist",
        {
            k: w["selections"][k].get("matches_historical_m3_selection")
            for k in ("independent", "correlation_aware", "correlation_aware_confidence")
        },
    )
    print()
