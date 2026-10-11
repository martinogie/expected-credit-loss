import unittest

import numpy as np
import pandas as pd

from ecl.analysis import quarter_end, roll_forward
from ecl.data import simulate
from ecl.engine import project, snapshot, stage
from ecl.macro import scenarios
from ecl.model import TransitionModel, estimate_lgd


class Book:
    loans, panel, rec, unemp = simulate()
    T = len(unemp) - 1
    model = TransitionModel(sample=150_000).fit(panel, loans, T - 3)
    lgd, _ = estimate_lgd(loans, rec, T - 3)
    snap = snapshot(loans, panel, T)
    sample = snap.sample(1500, random_state=0)


class TestModel(unittest.TestCase):
    def test_transition_rows_sum_to_one(self):
        s = Book.sample
        for state in ("C", "D30", "D60", "D90"):
            P = Book.model.proba(state, s["grade"], s["mob"], 4.5, s["months_since_dq"])
            np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)

    def test_lgd_is_a_fraction(self):
        self.assertTrue(((Book.lgd > 0.5) & (Book.lgd < 1)).all())

    def test_riskier_grades_have_higher_pd(self):
        r = project(Book.sample, Book.model, Book.lgd, np.full(72, 4.6))
        pd_by = r["pd12"].groupby(Book.sample["grade"]).mean()
        self.assertTrue(pd_by.is_monotonic_increasing)


class TestEngine(unittest.TestCase):
    def test_worse_economy_needs_more_reserve(self):
        paths, _ = scenarios()
        ecl = {k: project(Book.sample, Book.model, Book.lgd, p)["ecl_life"].sum() for k, p in paths.items()}
        self.assertLess(ecl["Upside"], ecl["Base"])
        self.assertLess(ecl["Base"], ecl["Downside"])
        self.assertLess(ecl["Downside"], ecl["Severe"])

    def test_lifetime_exceeds_twelve_month(self):
        r = project(Book.sample, Book.model, Book.lgd, np.full(72, 4.6))
        self.assertTrue((r["ecl_life"] >= r["ecl12"] - 1e-9).all())
        self.assertTrue((r["pd_life"] <= 1 + 1e-9).all())

    def test_staging_rules(self):
        s = Book.sample
        st = stage(s, pd.Series(0.01, index=s.index), pd.Series(0.01, index=s.index))
        self.assertTrue((st[s["state"] == "D90"] == 3).all())
        self.assertTrue((st[s["state"].isin(["D30", "D60"])] == 2).all())
        self.assertTrue((st[s["state"] == "C"] == 1).all())
        doubled = stage(s, pd.Series(0.03, index=s.index), pd.Series(0.01, index=s.index))
        self.assertTrue((doubled[s["state"] == "C"] == 2).all())


class TestRollForward(unittest.TestCase):
    def test_roll_forward_reconciles(self):
        L, P, R, U = Book.loans, Book.panel, Book.rec, Book.unemp
        keep = set(Book.loans.sample(8000, random_state=1)["loan_id"])
        loans = L[L["loan_id"].isin(keep)]
        panel = P[P["loan_id"].isin(keep)]
        q2 = quarter_end(loans, panel, Book.model, Book.lgd, Book.T - 3, "2026-06", U)
        q3 = quarter_end(loans, panel, Book.model, Book.lgd, Book.T, "2026-09", U)
        for basis in ("cecl", "ifrs9"):
            _, _, check = roll_forward(loans.reset_index(drop=True), R, Book.model, Book.lgd, q2, q3, basis)
            self.assertAlmostEqual(check, 0.0, delta=1.0)
        self.assertLessEqual(q3[3]["ifrs9"].sum(), q3[3]["cecl"].sum())


if __name__ == "__main__":
    unittest.main()
