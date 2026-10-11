"""Synthetic consumer installment-loan book, January 2021 to September 2026.

Each month every open loan moves between delinquency states:

    C (current) -> D30 -> D60 -> D90 -> CO (charged off at 120+ days)
    and C -> PP (prepaid in full); a loan that pays down to zero is MAT (matured).

The true monthly transition odds depend on the loan's grade, its age (credit losses peak in the
first 9-18 months), the unemployment rate, whether the borrower was delinquent in the last six
months (cured borrowers re-default far more often), and a hidden borrower-level frailty the
models never see. Charged-off loans recover 10-25% of balance over the following 12 months.
"""
import numpy as np
import pandas as pd

from .macro import AS_OF, START, history

GRADES = ["A", "B", "C", "D", "E"]
STATES = ["C", "D30", "D60", "D90", "CO", "PP", "MAT"]
OPEN = ["C", "D30", "D60", "D90"]
GRADE_SPEC = {  # base monthly C->D30 odds, APR, amount range, recovery rate, mix 2021-24, mix 2025+
    "A": (0.0035, 0.095, (8e3, 30e3), 0.25, 0.22, 0.17),
    "B": (0.0070, 0.130, (6e3, 25e3), 0.21, 0.30, 0.26),
    "C": (0.0120, 0.170, (5e3, 20e3), 0.17, 0.25, 0.26),
    "D": (0.0200, 0.220, (3e3, 15e3), 0.14, 0.15, 0.19),
    "E": (0.0320, 0.280, (2e3, 10e3), 0.11, 0.08, 0.12),
}
TERMS = [24, 36, 48, 60]


def seasoning(mob):
    """Delinquency risk by months on book: low at first, peaking around a year, then easing."""
    return 0.45 + 0.95 * np.exp(-((mob - 12) / 9.0) ** 2) + 0.25 * (mob > 12)


def annuity_payment(principal, rate, n):
    r = rate / 12
    return principal * r / (1 - (1 + r) ** -n)


def originations(rng, months):
    rows = []
    lid = 0
    for i, m in enumerate(months):
        n = int(650 * (1.012 ** i) * rng.lognormal(0, 0.06))
        late = m >= pd.Period("2025-01", "M")
        mix = np.array([GRADE_SPEC[g][5 if late else 4] for g in GRADES])
        grades = rng.choice(GRADES, size=n, p=mix / mix.sum())
        for g in grades:
            lo, hi = GRADE_SPEC[g][2]
            amount = round(rng.uniform(lo, hi), -2)
            term = int(rng.choice(TERMS, p=[0.2, 0.4, 0.25, 0.15]))
            rate = GRADE_SPEC[g][1] + rng.normal(0, 0.008)
            rows.append((lid, str(m), i, g, amount, term, round(rate, 4)))
            lid += 1
    df = pd.DataFrame(rows, columns=["loan_id", "orig_month", "orig_idx", "grade", "amount", "term", "rate"])
    df["payment"] = annuity_payment(df["amount"], df["rate"], df["term"])
    return df


def true_probs(state, grade_base, mob, u, frailty, recent):
    """Return an (n, 7) matrix of next-state probabilities for loans currently in `state`."""
    n = len(mob)
    p = np.zeros((n, len(STATES)))
    macro = np.exp(0.18 * (u - 4.0))
    if state == "C":
        d = np.clip(grade_base * seasoning(mob) * macro * frailty * np.where(recent, 2.6, 1.0), 0, 0.5)
        pp = 0.016 * (1.2 - 0.1 * np.searchsorted([0.004, 0.008, 0.013, 0.021], grade_base))
        p[:, 1], p[:, 5] = d, pp
        p[:, 0] = 1 - d - pp
    elif state == "D30":
        roll = np.clip(0.33 * macro ** 0.6 * frailty ** 0.3, 0, 0.8)
        p[:, 2], p[:, 0], p[:, 5] = roll, 0.50 - 0.3 * (roll - 0.33), 0.01
        p[:, 1] = 1 - p[:, 2] - p[:, 0] - p[:, 5]
    elif state == "D60":
        roll = np.clip(0.58 * macro ** 0.4, 0, 0.9)
        p[:, 3], p[:, 0] = roll, 0.24 - 0.3 * (roll - 0.58)
        p[:, 2] = 1 - p[:, 3] - p[:, 0]
    elif state == "D90":
        co = np.clip(0.84 * macro ** 0.2, 0, 0.97)
        p[:, 4], p[:, 0] = co, 0.09
        p[:, 3] = 1 - p[:, 4] - p[:, 0]
    return np.clip(p, 0, 1) / np.clip(p, 0, 1).sum(axis=1, keepdims=True)


def simulate(seed=21):
    rng = np.random.default_rng(seed)
    unemp = history()
    months = unemp.index
    loans = originations(rng, months)
    n = len(loans)
    base = loans["grade"].map({g: s[0] for g, s in GRADE_SPEC.items()}).to_numpy()
    rec_rate = loans["grade"].map({g: s[3] for g, s in GRADE_SPEC.items()}).to_numpy()
    frailty = rng.lognormal(-0.08, 0.4, n)
    state = np.full(n, -1)                       # -1 = not yet originated
    bal = loans["amount"].to_numpy().astype(float)
    pmt = loans["payment"].to_numpy()
    r = loans["rate"].to_numpy() / 12
    orig = loans["orig_idx"].to_numpy()
    co_month = np.full(n, -1)
    co_balance = np.zeros(n)
    exit_month = np.full(n, -1)
    last_dq = np.full(n, -1000)
    panel = []

    for t in range(len(months)):
        u = unemp.iloc[t]
        new = orig == t
        state[new] = 0
        active = np.isin(state, [0, 1, 2, 3]) & ~new
        idx = np.flatnonzero(active)
        if len(idx):
            mob = t - orig[idx]
            nxt = np.empty(len(idx), dtype=int)
            for s in range(4):
                sel = state[idx] == s
                if not sel.any():
                    continue
                recent = (t - last_dq[idx][sel]) <= 6          # cured borrowers often re-default
                P = true_probs(STATES[s], base[idx][sel], mob[sel], u, frailty[idx][sel], recent)
                cum = P.cumsum(axis=1)
                draw = rng.random(sel.sum())[:, None]
                nxt[sel] = (draw > cum).sum(axis=1)
            bal_before = bal[idx].copy()
            paying = nxt == 0
            bal[idx[paying]] = np.maximum(bal[idx[paying]] * (1 + r[idx[paying]]) - pmt[idx[paying]], 0.0)
            matured = paying & (bal[idx] <= 1.0)
            nxt[matured] = 6
            prepaid = nxt == 5
            bal[idx[prepaid]] = 0.0
            charged = nxt == 4
            co_month[idx[charged]] = t
            co_balance[idx[charged]] = bal_before[charged]
            exit_month[idx[nxt >= 4]] = t
            panel.append(np.column_stack([idx, np.full(len(idx), t), mob, state[idx], nxt, bal_before, bal[idx]]))
            state[idx] = nxt
            last_dq[idx[(nxt >= 1) & (nxt <= 3)]] = t

    panel = pd.DataFrame(np.vstack(panel), columns=["loan_id", "month_idx", "mob", "from_state", "to_state", "balance", "balance_end"])
    for c in ("loan_id", "month_idx", "mob", "from_state", "to_state"):
        panel[c] = panel[c].astype(int)
    panel["month"] = months[panel["month_idx"]].astype(str)
    panel["unemployment"] = unemp.to_numpy()[panel["month_idx"]]
    # behavioural signal: months since the loan was last 30+ days past due (99 = never)
    panel = panel.sort_values(["loan_id", "month_idx"], kind="stable").reset_index(drop=True)
    dq_month = panel["month_idx"].where(panel["to_state"].between(1, 3))
    last_dq = dq_month.groupby(panel["loan_id"]).ffill().groupby(panel["loan_id"]).shift(1)
    panel["months_since_dq"] = (panel["month_idx"] - last_dq).fillna(99).clip(upper=99).astype(int)

    # recoveries: 12 monthly instalments after charge-off, front-loaded, with noise
    rec = []
    w = 0.82 ** np.arange(12)
    w /= w.sum()
    for i in np.flatnonzero(co_month >= 0):
        total = co_balance[i] * np.clip(rng.normal(rec_rate[i], 0.06), 0, 0.6)
        for k in range(12):
            m = co_month[i] + 1 + k
            if m < len(months):
                rec.append((i, m, total * w[k]))
    recoveries = pd.DataFrame(rec, columns=["loan_id", "month_idx", "amount"])

    loans["state"] = [STATES[s] if s >= 0 else "NA" for s in state]
    loans["balance"] = np.where(np.isin(state, [0, 1, 2, 3]), bal, 0.0)
    loans["co_month_idx"] = co_month
    loans["co_balance"] = co_balance
    loans["exit_month_idx"] = exit_month
    return loans, panel, recoveries, unemp
