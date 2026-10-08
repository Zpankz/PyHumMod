"""Tests for the architecture database. Run with: python -m unittest discover tests"""

import ast
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archdb import extract  # noqa: E402
from archdb.query import ArchDB  # noqa: E402


class ArchDBTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "arch.db")
        extract.build(extract.DEFAULT_SOURCE, cls.path)
        cls.db = ArchDB(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.db.con.close()
        cls.tmp.cleanup()

    def test_every_live_class_is_a_module(self):
        with open(extract.DEFAULT_SOURCE, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        modules = {r["name"] for r in self.db.sql("SELECT name FROM modules WHERE name <> 'hummod'")}
        self.assertEqual(classes, modules)

    def test_every_assignment_outside_init_is_an_equation(self):
        with open(extract.DEFAULT_SOURCE, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        live = {}
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                live[n.name] = n
        expected = 0
        for cls in live.values():
            for m in cls.body:
                if isinstance(m, ast.FunctionDef) and m.name != "__init__" and not m.name.endswith("_curve"):
                    expected += sum(isinstance(s, (ast.Assign, ast.AugAssign)) for s in ast.walk(m))
        step = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "step")
        expected += sum(isinstance(s, (ast.Assign, ast.AugAssign)) for s in ast.walk(step))
        got = self.db.sql("SELECT COUNT(*) AS n FROM equations "
                          "WHERE kind NOT IN ('implicit_iterate', 'implicit_residual')")[0]["n"]
        # implicit_residual rows come from `return` statements, not assignments
        self.assertEqual(expected, got)

    def test_state_variable_and_its_rate(self):
        v = self.db.variable("ADHPool.Mass")
        self.assertEqual(v["role"], "state")
        self.assertEqual(v["integrator"], "diffeq")
        integrate = [e for e in v["defined_by"] if e["kind"] == "integrate"]
        self.assertEqual(integrate[0]["derivative"], "ADHPool.Change")

    def test_direct_dependencies(self):
        inputs = {r["variable"] for r in self.db.inputs("ADHPool.conc_ADH")}
        self.assertEqual(inputs, {"ADHPool.Mass", "ECFV.Vol_L"})
        outputs = {r["variable"] for r in self.db.outputs("ADHPool.conc_ADH")}
        self.assertIn("ADHClearance.Kidney", outputs)

    def test_lagged_read_of_state_updated_later_in_step(self):
        # ADHSecretion.Base reads ADHFastMass.Mass before ADHFastMass integrates it.
        row = self.db.sql("SELECT i.lagged FROM equation_inputs i JOIN equations e ON e.id = i.equation_id "
                          "WHERE e.target = 'ADHSecretion.Base' AND i.variable = 'ADHFastMass.Mass'")
        self.assertEqual(row[0]["lagged"], 1)

    def test_entailment_path(self):
        chain = self.db.path("BetaBlockade.Block_percent", "Heart_Ventricles.Rate")
        self.assertEqual(chain[0], "BetaBlockade.Block_percent")
        self.assertEqual(chain[-1], "Heart_Ventricles.Rate")
        down = {r["variable"] for r in self.db.downstream("BetaBlockade.Block_percent")}
        self.assertTrue(set(chain[1:]) <= down)

    def test_schedule_follows_step(self):
        top = self.db.sql("SELECT function FROM execution WHERE depth = 0 AND item = 'call' ORDER BY ord")
        self.assertEqual([r["function"] for r in top][:4], [
            "Structure.Context_func", "Structure.Parms_func", "Structure.Dervs_func", "Structure.Wrapup_func"])
        conditional = self.db.sql("SELECT condition FROM execution WHERE function = 'Ovaries.Parms_func'")
        self.assertEqual(conditional[0]["condition"], "Gender.IsFemale")

    def test_implicit_equation_links_through_residual(self):
        up = {r["variable"] for r in self.db.upstream("Skin_Flow.PO2", 3)}
        self.assertIn("Skin_Flow.Calc_func.PO2implicitfunc", up)
        self.assertIn("HgbTissue.pO2", up)

    def test_curve_matches_hermite_spline(self):
        db = self.db
        self.assertEqual(db.eval_curve("ADHSecretion.NeuralEffect_curve", 1.0), 1.0)
        self.assertEqual(db.eval_curve("ADHSecretion.NeuralEffect_curve", 0.0), 0.4)   # clamped
        self.assertEqual(db.eval_curve("ADHSecretion.NeuralEffect_curve", 9.0), 20.0)  # clamped
        mid = db.eval_curve("ADHSecretion.NeuralEffect_curve", 1.1)
        self.assertTrue(1.0 < mid < 2.0)

    def test_known_conversion_defects_are_reported(self):
        kinds = {r["kind"] for r in self.db.sql("SELECT DISTINCT kind FROM issues")}
        self.assertIn("duplicate_class", kinds)
        undeclared = {r["subject"] for r in self.db.issues("undeclared_variable")}
        self.assertIn("ADHPool.InitialConc", undeclared)

    def test_build_is_deterministic(self):
        other = os.path.join(self.tmp.name, "again.db")
        extract.build(extract.DEFAULT_SOURCE, other)
        con_a, con_b = sqlite3.connect(self.path), sqlite3.connect(other)
        try:
            self.assertEqual(list(con_a.iterdump()), list(con_b.iterdump()))
        finally:
            con_a.close()
            con_b.close()


if __name__ == "__main__":
    unittest.main()
