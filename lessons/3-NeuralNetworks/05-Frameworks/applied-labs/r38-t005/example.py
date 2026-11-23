"""
Applied Lab R38-T005: Agent-Component Routing — Offline Service Adapter
Lesson: lessons/3-NeuralNetworks/05-Frameworks

Contract:
- Route incoming requests to named handlers using explicit intent rules:
  * Contains 'help' -> 'support'
  * Contains 'calculate' -> 'calculator'
  * Any other / unrecognized input -> 'fallback'
- Ambiguous intent priority:
  * When multiple intents match (e.g., both 'help' and 'calculate'), prioritize 'support'
    over 'calculator' ('support' > 'calculator' > 'fallback').
  * Generate an auditable decision record logging the chosen rule and decision rationale.
- Service Adapter Pattern:
  * Handlers implement a callable interface.
  * Deterministic local adapter mocks the remote cloud services.
  * Explicitly note: This mock does NOT establish network latency, token auth, rate-limiting,
    or distributed consensus.
"""

from typing import Dict, Any, List, Protocol, Callable


class ServiceHandler(Protocol):
    """Callable protocol for routing handlers."""
    def __call__(self, request: str) -> str:
        ...


class SupportServiceAdapter:
    """Deterministic local adapter for support service."""
    def __call__(self, request: str) -> str:
        return f"SupportService: Assistance provided for '{request}'."


class CalculatorServiceAdapter:
    """Deterministic local adapter for calculation service."""
    def __call__(self, request: str) -> str:
        return f"CalculatorService: Calculation handled for '{request}'."


class FallbackServiceAdapter:
    """Deterministic fallback handler for unknown intents."""
    def __call__(self, request: str) -> str:
        return f"FallbackService: Default resolution for '{request}'."


class ComponentRouter:
    """
    Routes requests to adapters according to explicit intent keywords and documented priority.
    """
    PRIORITY = ["support", "calculator"]

    def __init__(self):
        self.handlers: Dict[str, Callable[[str], str]] = {
            "support": SupportServiceAdapter(),
            "calculator": CalculatorServiceAdapter(),
            "fallback": FallbackServiceAdapter(),
        }
        self.audit_log: List[Dict[str, Any]] = []

    def route_request(self, request: str) -> Dict[str, Any]:
        """
        Routes the request, logs the audit record, and executes the chosen adapter.
        """
        if not isinstance(request, str):
            raise TypeError(f"Request must be a string, got {type(request).__name__}")
        
        normalized = request.lower()
        matched: List[str] = []
        if "help" in normalized:
            matched.append("support")
        if "calculate" in normalized:
            matched.append("calculator")

        if len(matched) == 0:
            selected = "fallback"
            reason = "No known intent matched; routed to default fallback."
        elif len(matched) == 1:
            selected = matched[0]
            reason = f"Single intent '{selected}' matched."
        else:
            # Ambiguous: multiple intents matched. Select by highest priority.
            for p in self.PRIORITY:
                if p in matched:
                    selected = p
                    break
            reason = f"Ambiguous match {matched}; resolved by priority rule: {' > '.join(self.PRIORITY)}."

        handler = self.handlers[selected]
        response = handler(request)

        record = {
            "request": request,
            "matched_intents": matched,
            "selected_handler": selected,
            "priority_reason": reason,
            "response": response,
            "real_service_limitations": [
                "No live network latency simulation",
                "No authentication/token refresh",
                "No remote error code propagation"
            ]
        }
        self.audit_log.append(record)
        return record


if __name__ == "__main__":
    router = ComponentRouter()
    print("Help request:", router.route_request("Please help me"))
    print("Calculate request:", router.route_request("calculate 2 + 2"))
    print("Ambiguous request:", router.route_request("Help me calculate this"))
    print("Unknown request:", router.route_request("Random gibberish"))
