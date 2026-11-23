"""
Applied Lab R38-T004: Bounding-Box Intersection & IoU with State Trace
Lesson: lessons/4-ComputerVision/11-ObjectDetection

Contract:
- Coordinate convention: [x1, y1, x2, y2] where (x1, y1) is top-left and (x2, y2) is bottom-right.
  Requires x1 <= x2 and y1 <= y2. Inverted boxes (x1 > x2 or y1 > y2) are rejected with ValueError.
- Boxes [0, 0, 2, 2] and [1, 1, 3, 3] yield IoU = 1/7 (~0.142857).
- Identical non-zero boxes yield IoU = 1.0.
- Disjoint boxes yield IoU = 0.0.
- Focus: Record intermediate states (box areas, intersection coordinates, intersection area,
  union area) and verify the fundamental invariant: Union = Area(A) + Area(B) - Intersection.
"""

from typing import List, Tuple, Dict, Any, Union

Box = Union[List[float], Tuple[float, float, float, float]]


def validate_box(box: Box, name: str = "box") -> Tuple[float, float, float, float]:
    """Validates coordinate convention [x1, y1, x2, y2] and rejects inverted boxes."""
    if not isinstance(box, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of 4 numbers.")
    if len(box) != 4:
        raise ValueError(f"{name} must contain exactly 4 coordinates [x1, y1, x2, y2].")
    
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if x1 > x2 or y1 > y2:
        raise ValueError(f"Inverted {name}: coords [{x1}, {y1}, {x2}, {y2}] violate x1 <= x2 and y1 <= y2.")
    return (x1, y1, x2, y2)


def box_area(coords: Tuple[float, float, float, float]) -> float:
    """Computes area of axis-aligned box."""
    return (coords[2] - coords[0]) * (coords[3] - coords[1])


def compute_iou_with_trace(box_a: Box, box_b: Box) -> Dict[str, Any]:
    """
    Computes Intersection-over-Union (IoU) with explicit intermediate state trace and invariant verification.
    """
    a = validate_box(box_a, "box_a")
    b = validate_box(box_b, "box_b")

    area_a = box_area(a)
    area_b = box_area(b)

    # Intersection coordinates
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection_area = iw * ih

    union_area = area_a + area_b - intersection_area

    if union_area == 0.0:
        iou = 0.0
    else:
        iou = intersection_area / union_area

    # Invariant checks
    # 1. Inclusion-Exclusion: union_area == area_a + area_b - intersection_area
    inv_union = abs(union_area - (area_a + area_b - intersection_area)) < 1e-9
    # 2. IoU definition: iou * union_area == intersection_area
    inv_iou = abs(iou * union_area - intersection_area) < 1e-9
    invariant_holds = inv_union and inv_iou

    return {
        "box_a": list(a),
        "box_b": list(b),
        "area_a": area_a,
        "area_b": area_b,
        "intersection_coords": [ix1, iy1, ix2, iy2] if iw > 0 and ih > 0 else None,
        "intersection_area": intersection_area,
        "union_area": union_area,
        "iou": iou,
        "invariant_holds": invariant_holds
    }


if __name__ == "__main__":
    b1 = [0, 0, 2, 2]
    b2 = [1, 1, 3, 3]
    res = compute_iou_with_trace(b1, b2)
    print("Trace for [0,0,2,2] and [1,1,3,3]:", res)
