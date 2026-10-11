"""Fit the credit models from history.

Transition model: for each open delinquency state, a multinomial logistic regression for next
month's state, with grade, months on book, the unemployment rate and a behavioural flag (30+ days
past due in the last six months) as drivers. Chained month
by month, the four models give a lifetime probability of charge-off under any unemployment path.

LGD: 1 - present value of recoveries / balance at charge-off, by grade, using only charge-offs
old enough to have a full 12 months of recovery history.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data import GRADES, STATES

MOB_EDGES = [3, 6, 9, 12, 18, 24, 36]
TARGETS = {  # from-state -> next states modelled (MAT counts as paying, i.e. C)
    "C": ["C", "D30", "PP"],
    "D30": ["C", "D30", "D60"],
    "D60": ["C", "D60", "D90"],
    "D90": ["C", "D90", "CO"],
}


RECENT = 6   # months: a loan that was 30+ DPD this recently counts as recently delinquent


def features(grade, mob, u, msdq):
    grade = np.asarray(grade)
    mob = np.asarray(mob)
    cols = [(grade == g).astype(float) for g in GRADES[1:]]
    b = np.searchsorted(MOB_EDGES, mob, side="right")
    cols += [(b == k).astype(float) for k in range(1, len(MOB_EDGES) + 1)]
    cols.append(np.broadcast_to(np.asarray(u, float), mob.shape).astype(float) - 4.0)
    cols.append((np.asarray(msdq) <= RECENT).astype(float))
    return np.column_stack(cols)


class TransitionModel:
    def __init__(self, sample=400_000, seed=0):
        self.sample, self.seed = sample, seed
        self.models = {}

    def fit(self, panel, loans, last_month_idx):
        rng = np.random.default_rng(self.seed)
        p = panel[panel["month_idx"] <= last_month_idx]
        grade = loans["grade"].to_numpy()[p["loan_id"].to_numpy()]
        to = p["to_state"].map(lambda s: STATES[s]).replace({"MAT": "C"}).to_numpy()
        frm = p["from_state"].to_numpy()
        for si, s in enumerate(["C", "D30", "D60", "D90"]):
            sel = np.flatnonzero(frm == si)
            if s == "C" and len(sel) > self.sample:
                sel = rng.choice(sel, self.sample, replace=False)
            y = to[sel]
            y = np.where(np.isin(y, TARGETS[s]), y, "C")       # rare exits (e.g. prepay from D60) -> paying
            X = features(grade[sel], p["mob"].to_numpy()[sel], p["unemployment"].to_numpy()[sel],
                         p["months_since_dq"].to_numpy()[sel])
            m = LogisticRegression(max_iter=1000, C=10.0)
            m.fit(X, y)
            self.models[s] = m
        return self

    def proba(self, state, grade, mob, u, msdq):
        """(n, 7) next-state probabilities in STATES order."""
        m = self.models[state]
        P = m.predict_proba(features(grade, mob, u, msdq))
        out = np.zeros((len(P), len(STATES)))
        for j, c in enumerate(m.classes_):
            out[:, STATES.index(c)] = P[:, j]
        return out


def estimate_lgd(loans, recoveries, last_month_idx, follow_up=12):
    """LGD by grade from charge-offs with a complete recovery window."""
    co = loans[(loans["co_month_idx"] >= 0) & (loans["co_month_idx"] <= last_month_idx - follow_up)]
    rec = recoveries[recoveries["month_idx"] <= last_month_idx].merge(
        co[["loan_id", "co_month_idx", "rate"]], on="loan_id")
    rec["pv"] = rec["amount"] / (1 + rec["rate"] / 12) ** (rec["month_idx"] - rec["co_month_idx"])
    pv = rec.groupby("loan_id")["pv"].sum().reindex(co["loan_id"]).fillna(0.0).to_numpy()
    co = co.assign(lgd=1 - pv / co["co_balance"].to_numpy())
    by = co.groupby("grade").apply(lambda g: np.average(g["lgd"], weights=g["co_balance"]), include_groups=False)
    return by.reindex(GRADES), co.groupby("grade").size().reindex(GRADES)
