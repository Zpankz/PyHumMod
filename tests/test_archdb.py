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
                          "WHERE kind NOT IN ('implicit_iterate', 'implicit_residual', 'timer_count')")[0]["n"]
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
        # outside its points the curve continues along the end slope
        c = db.curve("ADHSecretion.NeuralEffect_curve")
        self.assertAlmostEqual(db.eval_curve("ADHSecretion.NeuralEffect_curve", 0.0),
                               c["y"][0] + c["slope"][0] * (0.0 - c["x"][0]))
        self.assertAlmostEqual(db.eval_curve("ADHSecretion.NeuralEffect_curve", 9.0),
                               c["y"][-1] + c["slope"][-1] * (9.0 - c["x"][-1]))
        mid = db.eval_curve("ADHSecretion.NeuralEffect_curve", 1.1)
        self.assertTrue(1.0 < mid < 2.0)

    def test_source_has_no_conversion_defects(self):
        # Only blocks step() never reaches remain, each with the reason it is unscheduled.
        kinds = {r["kind"] for r in self.db.sql("SELECT DISTINCT kind FROM issues")}
        self.assertEqual(kinds, {"unscheduled_block"})
        reasons = {r["detail"] for r in self.db.issues("unscheduled_block")}
        self.assertIn("not reachable from step() (event handler)", reasons)

    def test_conversion_defects_are_reported(self):
        source = """
class System:
    def __init__(self):
        self.Dx = 0.1

class Timer:
    def __init__(self, val, state, Dx):
        self.val = val

class Pool:
    def __init__(self):
        self.Mass = 1.0

class Pool:
    def __init__(self):
        self.Mass = 1.0
        self.Timer = Timer(0.0, "OFF", System.Dx)

    def Dervs_func(self):
        self.Mass = self.InitialConc * self.float("inf")
        if Timer < self.Mass:
            Other.Missing_func()

System = System()
Pool = Pool()

def step():
    Pool.Dervs_func()
"""
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "model.py")
            with open(src, "w") as f:
                f.write(source)
            extractor = extract.Extractor(src)
            extractor.run()
        found = {(kind, subject) for kind, subject, _detail, _line in extractor.issues}
        self.assertIn(("duplicate_class", "Pool"), found)
        self.assertIn(("undeclared_variable", "Pool.InitialConc"), found)
        self.assertIn(("missing_method", "self.float"), found)
        self.assertIn(("class_as_value", "Timer"), found)
        self.assertIn(("missing_block", "Other.Missing_func"), found)

    def test_timestep_is_an_input_of_integration(self):
        down = {r["variable"] for r in self.db.outputs("System.Dx")}
        self.assertIn("ADHPool.Mass", down)

    def test_call_guard_controls_callee_outputs(self):
        # ExcessLungWater.Failed_func runs only when OtherTissue_Function.Failed is true.
        up = {r["variable"] for r in self.db.inputs("ExcessLungWater.Grad")}
        self.assertIn("OtherTissue_Function.Failed", up)

    def test_mixed_edge_survives_condition_filter(self):
        # Index = FLAT inside a branch guarded by FLAT: the rhs edge must survive.
        up = {r["variable"] for r in self.db.upstream("Heart_ECG.Index", 1, include_conditions=False)}
        self.assertIn("Heart_ECG.FLAT", up)

    def test_algebraic_loops_exclude_lagged_reads(self):
        feedback = self.db.loops("feedback", 1)[0]["size"]
        algebraic = self.db.loops("algebraic", 1)[0]["size"]
        self.assertLess(algebraic, feedback)

    def test_nested_guards_are_parenthesised(self):
        rows = self.db.sql("SELECT condition FROM equations WHERE condition LIKE '%or%and%' "
                           "AND function = 'Heart_Ventricles.Calc_func'")
        self.assertTrue(rows)
        for r in rows:
            self.assertTrue(r["condition"].startswith("("), r["condition"])

    def test_every_registered_timer_counts(self):
        timers = {r["target"] for r in self.db.sql("SELECT target FROM equations WHERE kind = 'timer_count'")}
        self.assertIn("DailyPlannerControl.WaitingTimer", timers)
        self.assertIn("Heart_VFib.ElapsedTime", timers)
        self.assertEqual(len(timers), 14)

    def test_search_treats_underscore_literally(self):
        hits = {r["qualname"] for r in self.db.search("CorpusLuteum_Growth")}
        self.assertNotIn("Ovaries_CorpusLuteum.Growth", hits)
        self.assertIn("CorpusLuteum_Growth", hits)

    def test_concurrent_builds_do_not_collide(self):
        from concurrent.futures import ProcessPoolExecutor
        out = os.path.join(self.tmp.name, "shared.db")
        with ProcessPoolExecutor(4) as pool:
            list(pool.map(extract.build, [extract.DEFAULT_SOURCE] * 4, [out] * 4))
        con = sqlite3.connect(out)
        try:
            self.assertGreater(con.execute("SELECT COUNT(*) FROM equations").fetchone()[0], 0)
        finally:
            con.close()
        leftovers = [f for f in os.listdir(self.tmp.name) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

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
