"""Expected credit loss projection, IFRS 9 staging and CECL / IFRS 9 allowances.

For each open loan the engine propagates the probability of being in each delinquency state month
by month under an unemployment path, and books

    ECL = sum over months t of  P(charge-off in t) x LGD(grade) x EAD(t) x discount factor(t)

EAD(t) is the scheduled balance three months before charge-off (borrowers stop paying when they
go delinquent); loans already delinquent keep today's balance. Cash flows are discounted at
each loan's own rate (effective interest rate).

CECL books lifetime ECL on every loan. IFRS 9 books 12-month ECL in Stage 1 and lifetime ECL in
Stage 2 (significant increase in credit risk) and Stage 3 (90+ days past due).
"""
import numpy as np
import pandas as pd

from .data import OPEN, STATES

HORIZON = 72
SICR_MULTIPLE, SICR_FLOOR = 2.0, 0.005


def snapshot(loans, panel, month_idx):
    """Open loans at the end of a month, with state, balance and months on book."""
    p = panel[panel["month_idx"] == month_idx][["loan_id", "to_state", "balance_end"]]
    p = p[p["to_state"] <= 3]
    snap = loans.set_index("loan_id").loc[p["loan_id"]].copy()
    snap["state"] = [STATES[s] for s in p["to_state"]]
    snap["balance"] = p["balance_end"].to_numpy()
    new = loans[loans["orig_idx"] == month_idx].set_index("loan_id").copy()
    new["state"] = "C"
    new["balance"] = new["amount"]
    snap = pd.concat([snap, new])
    snap = snap[snap["balance"] > 1]
    snap["mob"] = month_idx - snap["orig_idx"]
    hist = panel[(panel["month_idx"] <= month_idx) & panel["to_state"].between(1, 3)]
    last_dq = hist.groupby("loan_id")["month_idx"].max()
    snap["months_since_dq"] = np.minimum(month_idx - snap.index.map(last_dq).to_numpy(dtype=float, na_value=-99), 99)
    return snap.reset_index()


def project(snap, model, lgd, upath, horizon=HORIZON):
    """Return per-loan 12-month and lifetime PD and ECL under one unemployment path.

    upath: array of length >= horizon, or an (n, horizon) array of loan-specific paths.
    """
    n = len(snap)
    grade = snap["grade"].to_numpy()
    mob0 = snap["mob"].to_numpy()
    bal = snap["balance"].to_numpy()
    r = snap["rate"].to_numpy() / 12
    pmt = snap["payment"].to_numpy()
    lgd_v = lgd.reindex(grade).to_numpy()
    msdq0 = snap["months_since_dq"].to_numpy()
    state0 = snap["state"].map({s: i for i, s in enumerate(OPEN)}).to_numpy()
    P = np.zeros((n, 4))
    P[np.arange(n), state0] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        n_rem = np.where(bal * r < pmt, -np.log(1 - bal * r / pmt) / np.log(1 + r), 0)
    upath = np.asarray(upath, float)
    pd12 = np.zeros(n)
    pdl = np.zeros(n)
    ecl12 = np.zeros(n)
    ecll = np.zeros(n)
    for k in range(1, horizon + 1):
        u = upath[:, k - 1] if upath.ndim == 2 else upath[k - 1]
        mob = mob0 + k
        newP = np.zeros((n, 4))
        co = np.zeros(n)
        for s in range(4):
            mass = P[:, s]
            if not mass.any():
                continue
            # loans in a delinquent state, or curing from one, carry the recent-delinquency flag
            msdq = msdq0 + k if s == 0 else np.zeros(n)
            T = model.proba(OPEN[s], grade, mob, u, msdq)
            newP += mass[:, None] * T[:, :4]
            co += mass * T[:, STATES.index("CO")]
        matured = k > n_rem
        newP[matured, 0] = 0.0                          # paid-down loans leave the book
        P = newP
        kk = max(k - 3, 0)
        sched = np.maximum(bal * (1 + r) ** kk - pmt * ((1 + r) ** kk - 1) / np.where(r > 0, r, 1), 0)
        ead = np.where(state0 == 0, sched, bal)
        loss = co * lgd_v * ead / (1 + r) ** k
        pdl += co
        ecll += loss
        if k <= 12:
            pd12 += co
            ecl12 += loss
        if P.sum() < 1e-6:
            break
    return pd.DataFrame({"pd12": pd12, "pd_life": pdl, "ecl12": ecl12, "ecl_life": ecll}, index=snap.index)


def weighted(results, weights):
    out = sum(results[k] * w for k, w in weights.items())
    return out


def origination_pd12(snap, model, unemp_hist):
    """12-month PD each loan had at origination (unemployment at origination held flat)."""
    at_orig = snap.assign(mob=0, state="C", balance=snap["amount"], months_since_dq=99)
    u0 = unemp_hist.to_numpy()[snap["orig_idx"].to_numpy()]
    return project(at_orig, model, pd.Series(0.0, index=["A", "B", "C", "D", "E"]),
                   np.repeat(u0[:, None], 12, axis=1), horizon=12)["pd12"]


def stage(snap, pd12_now, pd12_orig):
    s = np.full(len(snap), 1)
    sicr = (pd12_now >= SICR_MULTIPLE * pd12_orig) & (pd12_now - pd12_orig >= SICR_FLOOR)
    s[sicr.to_numpy() | snap["state"].isin(["D30", "D60"]).to_numpy()] = 2
    s[snap["state"].eq("D90").to_numpy()] = 3
    return pd.Series(s, index=snap.index)


def allowance(snap, results, weights, stages):
    w = weighted(results, weights)
    ifrs9 = np.where(stages == 1, w["ecl12"], w["ecl_life"])
    return pd.DataFrame({"grade": snap["grade"], "state": snap["state"], "balance": snap["balance"],
                         "stage": stages, "pd12": w["pd12"], "pd_life": w["pd_life"],
                         "cecl": w["ecl_life"], "ifrs9": ifrs9}, index=snap.index)
