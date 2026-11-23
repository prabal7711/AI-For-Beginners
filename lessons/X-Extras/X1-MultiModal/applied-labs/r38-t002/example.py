"""
Applied Lab R38-T002: Classification Evaluation — Binary Metrics with Zero-Denominator Policy
Lesson: lessons/X-Extras/X1-MultiModal

Contract:
- Compute confusion matrix (TP, FN, FP, TN), precision, and recall for binary predictions.
- Canonical fixture: y_true=[1, 1, 0, 0], y_pred=[1, 0, 1, 0]
  yields TP=1, FN=1, FP=1, TN=1, precision=0.5, recall=0.5.
- Documented Zero-Denominator Policy:
  * When TP + FP == 0 (no positive predictions), precision is defined as 0.0.
  * When TP + FN == 0 (no actual positive labels), recall is defined as 0.0.
- Rejects non-binary, mismatched-length, or empty inputs.
"""

from typing import List, Dict, Union, Any

BinaryLabel = Union[int, bool]


def evaluate_binary_classification(
    y_true: List[BinaryLabel],
    y_pred: List[BinaryLabel],
    zero_division: float = 0.0
) -> Dict[str, Union[int, float]]:
    """
    Evaluates binary classification predictions against ground truth labels.
    
    Parameters:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        zero_division: Value returned when denominator is 0 (default: 0.0)
        
    Returns:
        Dict with TP, FN, FP, TN, precision, and recall.
        
    Raises:
        TypeError: If inputs are not lists or elements are not valid binary ints (0 or 1).
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if not isinstance(y_true, list) or not isinstance(y_pred, list):
        raise TypeError("y_true and y_pred must both be lists.")
    if len(y_true) == 0:
        raise ValueError("Inputs cannot be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        # Validate binary integer (0 or 1)
        if isinstance(yt, bool):
            yt = int(yt)
        if isinstance(yp, bool):
            yp = int(yp)
        if yt not in (0, 1):
            raise ValueError(f"Invalid label in y_true at index {i}: {yt}. Must be 0 or 1.")
        if yp not in (0, 1):
            raise ValueError(f"Invalid label in y_pred at index {i}: {yp}. Must be 0 or 1.")

        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        else:
            tn += 1

    # Precision = TP / (TP + FP) with zero-division policy
    if tp + fp == 0:
        precision = float(zero_division)
    else:
        precision = float(tp) / float(tp + fp)

    # Recall = TP / (TP + FN) with zero-division policy
    if tp + fn == 0:
        recall = float(zero_division)
    else:
        recall = float(tp) / float(tp + fn)

    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "precision": precision,
        "recall": recall
    }


if __name__ == "__main__":
    canonical_true = [1, 1, 0, 0]
    canonical_pred = [1, 0, 1, 0]
    metrics = evaluate_binary_classification(canonical_true, canonical_pred)
    print("Metrics for canonical fixture:", metrics)
