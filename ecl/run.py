"""End to end: simulate the book, fit models, compute Q2 and Q3 allowances, roll forward,
backtest, and write reports plus the project page.

    python -m ecl.run
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import backtest, quarter_end, roll_forward, vintage_curves
from .data import GRADES, simulate
from .macro import history, scenarios
from .model import TransitionModel, estimate_lgd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
SCEN = ["Upside", "Base", "Downside", "Severe"]


def segments(snap, res, alw):
    """ECL by grade x stage under each scenario, so the page can re-weight scenarios."""
    out = []
    for g in GRADES:
        for s in (1, 2, 3):
            m = (snap["grade"] == g).to_numpy() & (alw["stage"] == s).to_numpy()
            if not m.any():
                continue
            out.append({"grade": g, "stage": s, "loans": int(m.sum()), "balance": round(float(snap.loc[m, "balance"].sum())),
                        "ecl12": [round(float(res[k].loc[m, "ecl12"].sum())) for k in SCEN],
                        "ecl_life": [round(float(res[k].loc[m, "ecl_life"].sum())) for k in SCEN]})
    return out


def plot(unemp, paths, alw, rf, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    h = unemp[-36:]
    x0 = np.arange(-len(h) + 1, 1)
    ax[0].plot(x0, h.values, color="#0b0b0b", lw=2, label="History")
    colors = {"Upside": "#1baf7a", "Base": "#2a78d6", "Downside": "#eb6834", "Severe": "#e34948"}
    for k, p in paths.items():
        ax[0].plot(np.arange(0, 37), np.r_[h.values[-1], p[:36]], color=colors[k], lw=1.8, label=k)
    ax[0].set_title("Unemployment rate, % (months from Sep 2026)")
    ax[0].legend(frameon=False, fontsize=8)
    g = alw.groupby("grade")[["balance", "cecl", "ifrs9"]].sum()
    xx = np.arange(len(g))
    ax[1].bar(xx - 0.2, g["cecl"] / g["balance"] * 100, 0.4, color="#2a78d6", label="CECL (lifetime)")
    ax[1].bar(xx + 0.2, g["ifrs9"] / g["balance"] * 100, 0.4, color="#eb6834", label="IFRS 9 (staged)")
    ax[1].set_xticks(xx, g.index)
    ax[1].set_title("Allowance coverage by grade, % of balance")
    ax[1].legend(frameon=False, fontsize=8)
    vals = rf["amount"].to_numpy() / 1e6
    labels = [s.replace("Provision: ", "").replace(" (Jun 30)", "").replace(" (Sep 30)", "") for s in rf["item"]]
    run = 0
    for i, v in enumerate(vals):
        if i in (0, len(vals) - 1):
            ax[2].bar(i, v, color="#52514e", width=0.6)
            run = v if i == 0 else run
        else:
            ax[2].bar(i, v, bottom=run, color="#2a78d6" if v >= 0 else "#e34948", width=0.6)
            run += v
    ax[2].set_xticks(range(len(vals)), labels, rotation=35, ha="right", fontsize=7.5)
    ax[2].set_ylim(min(0, vals.min()) , vals[[0, -1]].max() * 1.25)
    ax[2].set_title("CECL allowance roll-forward, Q3 2026, $M")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    loans, panel, rec, unemp = simulate()
    T3 = len(unemp) - 1
    T2 = T3 - 3
    model = TransitionModel().fit(panel, loans, T2)
    lgd, lgd_n = estimate_lgd(loans, rec, T2)
    q2 = quarter_end(loans, panel, model, lgd, T2, "2026-06", unemp)
    q3 = quarter_end(loans, panel, model, lgd, T3, "2026-09", unemp)
    snap, res, weights, alw = q3
    rf_cecl, prov_c, chk_c = roll_forward(loans, rec, model, lgd, q2, q3, "cecl")
    rf_ifrs, prov_i, chk_i = roll_forward(loans, rec, model, lgd, q2, q3, "ifrs9")
    bt, bt_total = backtest(loans, panel, rec, unemp, T3 - 18)
    vint = vintage_curves(loans, panel)
    paths, _ = scenarios()

    bal = alw["balance"].sum()
    by_stage = alw.groupby("stage").agg(loans=("balance", "size"), balance=("balance", "sum"), ifrs9=("ifrs9", "sum"), cecl=("cecl", "sum"))
    by_grade = alw.groupby("grade").agg(loans=("balance", "size"), balance=("balance", "sum"), pd12=("pd12", "mean"),
                                        pd_life=("pd_life", "mean"), cecl=("cecl", "sum"), ifrs9=("ifrs9", "sum"))
    by_scen = {k: {"cecl": float(res[k]["ecl_life"].sum())} for k in SCEN}
    metrics = {
        "as_of": "2026-09-30", "loans": int(len(snap)), "balance": float(bal),
        "cecl": float(alw["cecl"].sum()), "ifrs9": float(alw["ifrs9"].sum()),
        "cecl_coverage": float(alw["cecl"].sum() / bal), "ifrs9_coverage": float(alw["ifrs9"].sum() / bal),
        "q2_cecl": float(q2[3]["cecl"].sum()), "q2_ifrs9": float(q2[3]["ifrs9"].sum()), "q2_balance": float(q2[3]["balance"].sum()),
        "scenario_cecl": by_scen, "weights": weights, "lgd": lgd.round(4).to_dict(), "lgd_obs": lgd_n.to_dict(),
        "provision_cecl": float(prov_c), "provision_ifrs9": float(prov_i), "rollforward_check": [float(chk_c), float(chk_i)],
        "backtest_total": bt_total, "runtime_seconds": round(time.time() - t0, 1),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    by_grade.round(4).to_csv(REPORTS / "allowance_by_grade.csv")
    by_stage.round(2).to_csv(REPORTS / "allowance_by_stage.csv")
    rf_cecl.round(0).to_csv(REPORTS / "rollforward_cecl.csv", index=False)
    rf_ifrs.round(0).to_csv(REPORTS / "rollforward_ifrs9.csv", index=False)
    bt.round(4).to_csv(REPORTS / "backtest_12m.csv")
    plot(unemp, paths, alw, rf_cecl, REPORTS / "results.png")

    hist = unemp[-36:]
    page = {
        "as_of": "Sep 30, 2026", "balance": round(float(bal)), "loans": int(len(snap)),
        "scenarios": SCEN, "weights": [weights[k] for k in SCEN],
        "paths": {k: [float(x) for x in paths[k][:36]] for k in SCEN},
        "history": {"labels": [str(p) for p in hist.index], "values": [float(x) for x in hist.values]},
        "segments": segments(snap, res, alw),
        "q2": {"cecl": round(metrics["q2_cecl"]), "ifrs9": round(metrics["q2_ifrs9"]), "balance": round(metrics["q2_balance"])},
        "rollforward": {"cecl": [[i, round(a)] for i, a in rf_cecl.itertuples(index=False)],
                        "ifrs9": [[i, round(a)] for i, a in rf_ifrs.itertuples(index=False)]},
        "backtest": [{"grade": g, **{k: (None if pd.isna(v) else round(float(v), 4)) for k, v in r.items()}} for g, r in bt.iterrows()],
        "backtest_total": {k: round(v, 4) for k, v in bt_total.items()},
        "vintages": {k: [round(x, 4) for x in v] for k, v in vint.items()},
        "lgd": {k: round(float(v), 3) for k, v in lgd.items()},
    }
    html = (ROOT / "web" / "template.html").read_text().replace("/*DATA*/null", json.dumps(page, separators=(",", ":")))
    (ROOT / "index.html").write_text(html)
    print(json.dumps({k: metrics[k] for k in ("loans", "balance", "cecl", "ifrs9", "cecl_coverage", "ifrs9_coverage",
                                               "q2_cecl", "provision_cecl", "rollforward_check", "backtest_total", "runtime_seconds")}, indent=1, default=float))
    print(rf_cecl.assign(amount=(rf_cecl["amount"] / 1e6).round(2)).to_string(index=False))
    print(by_stage.assign(**{c: (by_stage[c] / 1e6).round(2) for c in ("balance", "ifrs9", "cecl")}))
    print(bt.round(4))


if __name__ == "__main__":
    main()
