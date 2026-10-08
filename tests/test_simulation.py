"""Runs the simulator itself. Needs scipy; skipped without it.
Run with: python -m unittest discover tests"""

import importlib
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import scipy  # noqa: F401
except ImportError:  # pragma: no cover
    scipy = None


def module_values(hummod):
    for name, obj in vars(hummod).items():
        if type(obj).__module__ == hummod.__name__ and not isinstance(obj, type):
            for attr, value in vars(obj).items():
                yield name + "." + attr, value


@unittest.skipIf(scipy is None, "scipy is not installed")
class SimulationTest(unittest.TestCase):
    def setUp(self):
        # a fresh model each test: hummod keeps its state in module-level singletons
        sys.modules.pop("src.hummod", None)
        self.hummod = importlib.import_module("src.hummod")

    def run_steps(self, n):
        for _ in range(n):
            self.hummod.step()

    def assert_all_finite(self):
        for name, value in module_values(self.hummod):
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value), name)
            else:
                # numpy arrays leaking out of solvers used to crash PhGeneral.Calc_func
                self.assertNotEqual(type(value).__module__, "numpy", name)

    def test_steps_run_and_stay_physiological(self):
        self.run_steps(2000)  # 0.6 simulated minutes
        h = self.hummod
        self.assert_all_finite()
        self.assertTrue(55 < h.Heart_Ventricles.Rate < 90)
        self.assertTrue(80 < h.SystemicArtys.Pressure < 110)
        self.assertTrue(7.35 < h.BloodPh.ArtysPh < 7.48)

    def test_daily_dose_timer_runs(self):
        # these compared the Timer class itself with a float before the fix
        h = self.hummod
        for dose in (h.MidodrineDailyDose, h.DigoxinDailyDose, h.ThiazideDailyDose):
            dose.TakeDaily = True
        h.Pheochromocytoma.Switch = True
        self.run_steps(50)
        self.assert_all_finite()

    def test_event_handlers_run(self):
        h = self.hummod
        self.run_steps(10)
        for block in ("ADHPool.Initialize_func", "ANPPool.Initialize_func", "FSH.Initialize_func",
                      "LH.Initialize_func", "Heart_Defibrillator.ShockNow_func"):
            module, method = block.split(".")
            getattr(getattr(h, module), method)()
        self.run_steps(10)
        self.assert_all_finite()


if __name__ == "__main__":
    unittest.main()
