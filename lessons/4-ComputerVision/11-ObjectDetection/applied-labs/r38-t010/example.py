"""
Applied Lab R38-T010: Bounding-Box Intersection & Scenario Assessment
Lesson: lessons/4-ComputerVision/11-ObjectDetection

Contract:
- Coordinate convention: [x1, y1, x2, y2] where (x1, y1) is top-left and (x2, y2) is bottom-right.
  Requires x1 <= x2 and y1 <= y2. Inverted boxes (x1 > x2 or y1 > y2) are rejected with ValueError.
- Boxes [0, 0, 2, 2] and [1, 1, 3, 3] yield IoU = 1/7.
- Identical boxes yield IoU = 1.0; disjoint boxes yield IoU = 0.0.
- Focus: Three applied assessment questions with rationale and executable assertions:
  1. Prediction Question: Calculate IoU for scaled boxes [0, 0, 4, 4] and [2, 2, 6, 6] (expected 1/7).
  2. Failure Diagnosis Question: Diagnose why box [4, 5, 2, 8] is rejected (x1 > x2 inversion).
  3. Boundary Policy Question: Define and verify edge-touching boundary policy [0, 0, 2, 2] and [2, 0, 4, 2] (IoU = 0.0).
"""

from typing import List, Tuple, Dict, Any, Union

Box = Union[List[float], Tuple[float, float, float, float]]


def validate_box(box: Box, name: str = "box") -> Tuple[float, float, float, float]:
    """Validates 4-coordinate bounding box and rejects inverted boxes."""
    if not isinstance(box, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of 4 numbers.")
    if len(box) != 4:
        raise ValueError(f"{name} must contain exactly 4 coordinates [x1, y1, x2, y2].")

    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if x1 > x2 or y1 > y2:
        raise ValueError(f"Inverted {name}: coords [{x1}, {y1}, {x2}, {y2}] violate x1 <= x2 and y1 <= y2.")
    return (x1, y1, x2, y2)


def compute_iou(box_a: Box, box_b: Box) -> float:
    """Computes Intersection-over-Union (IoU) of two axis-aligned bounding boxes."""
    a = validate_box(box_a, "box_a")
    b = validate_box(box_b, "box_b")

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    union = area_a + area_b - intersection
    if union == 0.0:
        return 0.0
    return intersection / union


def scenario_assessment() -> Dict[str, Dict[str, Any]]:
    """
    Returns three applied scenario assessments with formal rationales and factual results.
    """
    # Question 1: Prediction
    box_p1 = [0, 0, 4, 4]
    box_p2 = [2, 2, 6, 6]
    iou_pred = compute_iou(box_p1, box_p2)
    q1 = {
        "question": "What is the IoU of boxes [0, 0, 4, 4] and [2, 2, 6, 6]?",
        "answer": iou_pred,
        "expected": 1.0 / 7.0,
        "rationale": (
            "Area(A) = 16, Area(B) = 16. Overlap rectangle is [2, 2, 4, 4] with area = 4. "
            "Union = 16 + 16 - 4 = 28. IoU = 4 / 28 = 1/7."
        )
    }

    # Question 2: Failure Diagnosis
    inverted_box = [4, 5, 2, 8]
    diag_error = ""
    try:
        validate_box(inverted_box)
    except ValueError as e:
        diag_error = str(e)
    q2 = {
        "question": "Why does box [4, 5, 2, 8] fail validation?",
        "diagnosed_error": diag_error,
        "rationale": "x1 (4) is greater than x2 (2), representing an inverted horizontal coordinate span."
    }

    # Question 3: Boundary Policy
    touching_a = [0, 0, 2, 2]
    touching_b = [2, 0, 4, 2]
    iou_touching = compute_iou(touching_a, touching_b)
    q3 = {
        "question": "What is the IoU for adjacent boxes sharing an edge [0, 0, 2, 2] and [2, 0, 4, 2]?",
        "answer": iou_touching,
        "expected": 0.0,
        "rationale": "The intersection width is max(0, 2 - 2) = 0. Therefore intersection area is 0, giving IoU = 0.0."
    }

    return {"q1": q1, "q2": q2, "q3": q3}


if __name__ == "__main__":
    canonical_iou = compute_iou([0, 0, 2, 2], [1, 1, 3, 3])
    print(f"Canonical IoU: {canonical_iou} (expected 1/7 = {1/7})")
    print("Scenario Assessments:", scenario_assessment())
