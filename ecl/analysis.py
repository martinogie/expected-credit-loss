"""Quarter-end allowance, Q2 -> Q3 roll-forward, and an out-of-time backtest."""
import numpy as np
import pandas as pd

from .data import GRADES
from .engine import allowance, origination_pd12, project, snapshot, stage
from .macro import history, scenarios
from .model import TransitionModel, estimate_lgd


def quarter_end(loans, panel, model, lgd, month_idx, as_of, unemp):
    snap = snapshot(loans, panel, month_idx)
    paths, weights = scenarios(as_of)
    res = {k: project(snap, model, lgd, p) for k, p in paths.items()}
    pd12_w = sum(res[k]["pd12"] * w for k, w in weights.items())
    st = stage(snap, pd12_w, origination_pd12(snap, model, unemp))
    return snap, res, weights, allowance(snap, res, weights, st)


def roll_forward(loans, recoveries, model, lgd, q2, q3, basis="cecl"):
    """Beginning allowance -> ending allowance with the provision split by driver."""
    snap2, res2, w2, a2 = q2
    snap3, res3, w3, a3 = q3
    m2, m3 = snap2["orig_idx"].max(), snap3["orig_idx"].max()
    a2i = a2.set_index(snap2["loan_id"])[basis]
    a3i = a3.set_index(snap3["loan_id"])[basis]
    ids2, ids3 = set(a2i.index), set(a3i.index)
    exited = loans.set_index("loan_id").loc[sorted(ids2 - ids3)]
    charged = exited.index[(exited["co_month_idx"] > m2) & (exited["co_month_idx"] <= m3)]
    paid = exited.index.difference(charged)
    new = sorted(ids3 - ids2)
    stay = sorted(ids2 & ids3)

    # macro effect: Q3 portfolio under Q2 scenarios vs Q3 scenarios
    paths2, _ = scenarios("2026-06")
    sel = snap3["loan_id"].isin(stay).to_numpy()
    s3 = snap3[sel]
    res_old = {k: project(s3, model, lgd, p) for k, p in paths2.items()}
    st = a3.loc[s3.index, "stage"]
    a_old = allowance(s3, res_old, w2, st).set_index(s3["loan_id"])[basis]

    gross_co = loans.set_index("loan_id").loc[charged, "co_balance"].sum()
    rec = recoveries[(recoveries["month_idx"] > m2) & (recoveries["month_idx"] <= m3)]["amount"].sum()
    begin, end = a2i.sum(), a3i.sum()
    comp = {
        "New originations": a3i.loc[new].sum(),
        "Paid off or matured (release)": -a2i.loc[paid].sum(),
        "Charge-offs net of reserve held": (gross_co - rec) - a2i.loc[charged].sum(),
        "Credit migration and seasoning": a_old.sum() - a2i.loc[stay].sum(),
        "Macroeconomic forecast": a3i.loc[stay].sum() - a_old.sum(),
    }
    rows = [("Beginning allowance (Jun 30)", begin), ("Gross charge-offs", -gross_co), ("Recoveries", rec)]
    rows += [(f"Provision: {k}", v) for k, v in comp.items()]
    rows += [("Ending allowance (Sep 30)", end)]
    check = begin - gross_co + rec + sum(comp.values()) - end
    return pd.DataFrame(rows, columns=["item", "amount"]), sum(comp.values()), check


def backtest(loans, panel, recoveries, unemp, month_idx, months_ahead=12):
    """Fit on data to `month_idx`, predict 12-month charge-offs for the book open then, and compare
    with what happened. Benchmark: each grade's trailing 24-month charge-off rate (a simple
    historical loss-rate method)."""
    model = TransitionModel().fit(panel, loans, month_idx)
    lgd, _ = estimate_lgd(loans, recoveries, month_idx)
    snap = snapshot(loans, panel, month_idx)
    actual_u = unemp.to_numpy()[month_idx + 1: month_idx + 1 + months_ahead]
    flat = np.full(months_ahead, unemp.iloc[month_idx])
    pred = project(snap, model, lgd, actual_u, horizon=months_ahead)
    pred_flat = project(snap, model, lgd, flat, horizon=months_ahead)
    co = snap["co_month_idx"].between(month_idx + 1, month_idx + months_ahead)
    # trailing loss-rate benchmark: share of loans open 24 months earlier that charged off within 12 months, by grade
    ref = snapshot(loans, panel, month_idx - 24)
    ref_rate = ref["co_month_idx"].between(month_idx - 23, month_idx - 12).groupby(ref["grade"]).mean()
    out = pd.DataFrame({
        "loans": snap.groupby("grade").size(),
        "actual": co.groupby(snap["grade"]).mean(),
        "model_actual_macro": pred["pd12"].groupby(snap["grade"]).mean(),
        "model_flat_macro": pred_flat["pd12"].groupby(snap["grade"]).mean(),
        "historical_rate": ref_rate,
    }).reindex(GRADES)
    total = {
        "actual": float(co.mean()), "model_actual_macro": float(pred["pd12"].mean()),
        "model_flat_macro": float(pred_flat["pd12"].mean()),
        "historical_rate": float((snap["grade"].map(ref_rate)).mean()),
    }
    return out, total


def vintage_curves(loans, panel, max_mob=48):
    """Cumulative charge-off rate by months on book for each annual vintage."""
    loans = loans.assign(vintage=loans["orig_month"].str[:4])
    last = panel["month_idx"].max()
    out = {}
    for v, g in loans.groupby("vintage"):
        co_mob = (g["co_month_idx"] - g["orig_idx"]).where(g["co_month_idx"] >= 0)
        age = last - g["orig_idx"]
        curve = []
        for m in range(0, max_mob + 1):
            seen = age >= m                      # only loans observed for at least m months
            if seen.sum() < 0.25 * len(g):
                break
            curve.append(float((co_mob[seen] <= m).mean()))
        if len(curve) > 6:
            out[v] = curve
    return out
