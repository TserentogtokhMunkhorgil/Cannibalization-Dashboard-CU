from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class ExpansionPageTests(unittest.TestCase):
    def test_page_startup_never_optimizes(self):
        root = Path(__file__).resolve().parents[1]
        with patch('expansion_optimization_page.completed', side_effect=AssertionError('Unrequested optimization')):
            app = AppTest.from_file(str(root / '05 Cannibalization Dashboard.py'), default_timeout=60).run()
            app.sidebar.selectbox[0].select('Expansion Optimization').run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.form_submit_button[0].label, 'Run optimization')

    def test_invalid_npv_is_reported_without_computation(self):
        root = Path(__file__).resolve().parents[1]
        with patch('expansion_optimization_page.completed', side_effect=AssertionError('Invalid NPV must not run')):
            app = AppTest.from_file(str(root / '05 Cannibalization Dashboard.py'), default_timeout=60).run()
            app.sidebar.selectbox[0].select('Expansion Optimization').run()
            app.session_state['expansion_objective'] = 'npv'
            app.run()
            app.form_submit_button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertIn('requires supplied cost', app.error[0].value)
