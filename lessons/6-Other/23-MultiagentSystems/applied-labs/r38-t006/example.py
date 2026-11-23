"""
Applied Lab R38-T006: Dependency-Plan Validation & Topological Ordering
Lesson: lessons/6-Other/23-MultiagentSystems

Contract:
- Validate and topologically order a task plan where tasks declare their dependencies.
- Fixture: A has no dependencies, B depends on A, C on B -> order ['A', 'B', 'C'].
- Reject unknown/missing dependencies: reporting the exact missing task ID.
- Reject cyclic dependencies: A -> B -> A is rejected with a cycle detection error.
- Refactoring Focus:
  1. Baseline implementation: monolithic validation & ordering function.
  2. Refactored architecture: distinct PlanValidator and TopologicalPlanner classes,
     proving strict before/after behavioral equivalence on identical fixtures.
"""

from typing import Dict, List, Set, Any


class MissingDependencyError(ValueError):
    """Raised when a task depends on an unknown task ID."""
    pass


class CyclicDependencyError(ValueError):
    """Raised when a dependency cycle is detected."""
    pass


def baseline_validate_and_order(plan: Dict[str, List[str]]) -> List[str]:
    """
    Baseline monolithic validation and topological sort.
    """
    if not isinstance(plan, dict):
        raise TypeError(f"Plan must be a dict, got {type(plan).__name__}")
    if len(plan) == 0:
        return []

    # 1. Validate dependencies exist
    all_tasks = set(plan.keys())
    for task_id, deps in plan.items():
        if not isinstance(deps, (list, tuple, set)):
            raise TypeError(f"Dependencies for '{task_id}' must be a sequence.")
        for dep in deps:
            if dep not in all_tasks:
                raise MissingDependencyError(f"Task '{task_id}' depends on unknown task ID '{dep}'")

    # 2. Compute in-degrees and adjacency
    in_degree = {task: 0 for task in all_tasks}
    dependents: Dict[str, List[str]] = {task: [] for task in all_tasks}

    for task, deps in plan.items():
        in_degree[task] = len(deps)
        for dep in deps:
            dependents[dep].append(task)

    # 3. Kahn's Algorithm
    ready = [task for task in sorted(all_tasks) if in_degree[task] == 0]
    ordered: List[str] = []

    while ready:
        curr = ready.pop(0)
        ordered.append(curr)
        for nxt in sorted(dependents[curr]):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                ready.append(nxt)

    if len(ordered) != len(all_tasks):
        raise CyclicDependencyError(f"Cyclic dependency detected in plan: ordered {ordered} of {all_tasks}")

    return ordered


class PlanValidator:
    """Extracted responsibility: Validates task presence and missing dependency IDs."""
    
    @staticmethod
    def validate(plan: Dict[str, List[str]]) -> None:
        if not isinstance(plan, dict):
            raise TypeError(f"Plan must be a dict, got {type(plan).__name__}")
        all_tasks = set(plan.keys())
        for task_id, deps in plan.items():
            if not isinstance(deps, (list, tuple, set)):
                raise TypeError(f"Dependencies for '{task_id}' must be a sequence.")
            for dep in deps:
                if dep not in all_tasks:
                    raise MissingDependencyError(f"Task '{task_id}' depends on unknown task ID '{dep}'")


class TopologicalPlanner:
    """Extracted responsibility: Computes deterministic execution order and detects cycles."""

    @staticmethod
    def order(plan: Dict[str, List[str]]) -> List[str]:
        all_tasks = set(plan.keys())
        in_degree = {task: 0 for task in all_tasks}
        dependents: Dict[str, List[str]] = {task: [] for task in all_tasks}

        for task, deps in plan.items():
            in_degree[task] = len(deps)
            for dep in deps:
                dependents[dep].append(task)

        ready = [t for t in sorted(all_tasks) if in_degree[t] == 0]
        ordered: List[str] = []

        while ready:
            curr = ready.pop(0)
            ordered.append(curr)
            for nxt in sorted(dependents[curr]):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    ready.append(nxt)

        if len(ordered) != len(all_tasks):
            raise CyclicDependencyError("Cyclic dependency detected in task plan.")

        return ordered


def refactored_validate_and_order(plan: Dict[str, List[str]]) -> List[str]:
    """Refactored pipeline delegating to PlanValidator and TopologicalPlanner."""
    if len(plan) == 0:
        return []
    PlanValidator.validate(plan)
    return TopologicalPlanner.order(plan)


if __name__ == "__main__":
    canonical_plan = {
        "A": [],
        "B": ["A"],
        "C": ["B"]
    }
    b_res = baseline_validate_and_order(canonical_plan)
    r_res = refactored_validate_and_order(canonical_plan)
    print("Baseline:", b_res)
    print("Refactored:", r_res)
    assert b_res == r_res
