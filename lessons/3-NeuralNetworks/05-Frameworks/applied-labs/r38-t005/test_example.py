"""
Unit tests for Applied Lab R38-T005: agent-component routing & offline service adapter.
"""

import unittest
from example import ComponentRouter, SupportServiceAdapter, CalculatorServiceAdapter, FallbackServiceAdapter


class TestComponentRouting(unittest.TestCase):

    def setUp(self):
        self.router = ComponentRouter()

    def test_help_routes_to_support(self):
        """help -> support"""
        rec = self.router.route_request("I need help with my account")
        self.assertEqual(rec["selected_handler"], "support")
        self.assertIn("support", rec["matched_intents"])
        self.assertIn("SupportService", rec["response"])

    def test_calculate_routes_to_calculator(self):
        """calculate -> calculator"""
        rec = self.router.route_request("Please calculate the loss")
        self.assertEqual(rec["selected_handler"], "calculator")
        self.assertIn("calculator", rec["matched_intents"])
        self.assertIn("CalculatorService", rec["response"])

    def test_unknown_routes_to_fallback(self):
        """unknown -> fallback"""
        rec = self.router.route_request("Check weather forecast")
        self.assertEqual(rec["selected_handler"], "fallback")
        self.assertEqual(rec["matched_intents"], [])
        self.assertIn("FallbackService", rec["response"])

    def test_ambiguous_intent_follows_priority(self):
        """Ambiguous intent (help + calculate) routes to support per documented priority."""
        rec = self.router.route_request("Can you help calculate total parameters?")
        self.assertEqual(rec["selected_handler"], "support")
        self.assertEqual(set(rec["matched_intents"]), {"support", "calculator"})
        self.assertIn("priority rule", rec["priority_reason"])

    def test_audit_record_logging(self):
        """All routing decisions must be recorded in the auditable log."""
        self.router.route_request("help")
        self.router.route_request("calculate")
        self.assertEqual(len(self.router.audit_log), 2)
        self.assertEqual(self.router.audit_log[0]["selected_handler"], "support")
        self.assertEqual(self.router.audit_log[1]["selected_handler"], "calculator")

    def test_invalid_input_type(self):
        """Non-string inputs raise TypeError."""
        with self.assertRaises(TypeError):
            self.router.route_request(12345)

    def test_deliberate_wrong_result_fails(self):
        """Deliberately wrong result triggers assertion failure."""
        rec = self.router.route_request("help")
        with self.assertRaises(AssertionError):
            self.assertEqual(rec["selected_handler"], "calculator")


if __name__ == "__main__":
    unittest.main()
