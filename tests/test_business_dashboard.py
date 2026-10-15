"""Smoke tests require the checked-in/frozen project data and generated planning outputs."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class BusinessDashboardTests(unittest.TestCase):
    def test_integrated_navigation_and_scenario_dashboard(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / 'outputs/planning/contributions.parquet').is_file():
            self.skipTest('Generate planning outputs before the dashboard integration smoke test.')
        app = AppTest.from_file(str(root / '05 Cannibalization Dashboard.py'), default_timeout=60).run()
        self.assertEqual(
            tuple(app.sidebar.selectbox[0].options),
            ("CU Network & Cannibalization", "Expansion Optimization", "Regression Results"),
        )
        self.assertFalse(app.exception)
        for section in (
            "Cannibalization Overview",
            "Planned New Store Locations",
            "Store-Level Exploration",
        ):
            app.segmented_control[0].select(section).run()
            self.assertFalse(app.exception, section)
        for page in ("Regression Results", "Expansion Optimization"):
            app.sidebar.selectbox[0].select(page).run()
            self.assertFalse(app.exception, page)
