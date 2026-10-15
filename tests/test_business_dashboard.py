"""Smoke tests require the checked-in/frozen project data and generated planning outputs."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class BusinessDashboardTests(unittest.TestCase):
    def test_four_pages_modes_outcomes_and_literal_search(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / 'outputs/planning/contributions.parquet').is_file():
            self.skipTest('Generate planning outputs before the dashboard integration smoke test.')
        app = AppTest.from_file(str(root / '05 Cannibalization Dashboard.py'), default_timeout=60).run()
        self.assertEqual(len(app.sidebar.selectbox[0].options), 4)
        self.assertFalse(app.exception)
        for page in ('Regression Results', 'Planned New Store Locations', 'Store-Level Exploration'):
            app.sidebar.selectbox[0].select(page).run()
            self.assertFalse(app.exception, page)
        app.sidebar.selectbox[2].select('customers').run()
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('raw').run()
        self.assertFalse(app.exception)
        app.sidebar.selectbox[1].select(400).run()
        self.assertFalse(app.exception)
        app.text_input[0].set_value('[').run()
        self.assertFalse(app.exception)
