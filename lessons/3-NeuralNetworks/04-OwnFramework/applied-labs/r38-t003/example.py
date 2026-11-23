"""
Applied Lab R38-T003: Single Perceptron Update — Comparative Implementation
Lesson: lessons/3-NeuralNetworks/04-OwnFramework

Contract:
- 2D input fixture: input=[1, 2], initial weights=[0, 0], bias=0, target=1, lr=1.0.
- Zero-score tie rule:
  score = w1*x1 + w2*x2 + b
  If score > 0 -> prediction = 1
  If score <= 0 -> prediction = 0  (explicit documented tie rule)
- For fixture:
  score = 0*1 + 0*2 + 0 = 0 -> initial prediction = 0
  error = target - prediction = 1 - 0 = 1
  w_new = w + lr * error * x = [0, 0] + 1.0 * 1 * [1, 2] = [1.0, 2.0]
  b_new = b + lr * error = 0 + 1.0 * 1 = 1.0
- Focus: Compare the direct algorithmic implementation with an independent closed-form
  calculation, proving exact equivalence on identical inputs.
"""

from typing import List, Tuple, Dict, Any


def direct_perceptron_step(
    weights: List[float],
    bias: float,
    x: List[float],
    target: int,
    learning_rate: float = 1.0
) -> Dict[str, Any]:
    """
    Direct perceptron prediction and update step.
    
    Zero-score tie rule: score <= 0 predicts 0, score > 0 predicts 1.
    """
    if len(weights) != len(x):
        raise ValueError(f"Weights dimension ({len(weights)}) must match input ({len(x)})")
    if target not in (0, 1):
        raise ValueError("Target must be 0 or 1")
    
    # Calculate score (weighted sum + bias)
    score = sum(w * xi for w, xi in zip(weights, x)) + bias
    
    # Apply zero-score tie rule
    initial_prediction = 1 if score > 0 else 0
    
    # Calculate error
    error = target - initial_prediction
    
    # Update weights and bias
    new_weights = [w + learning_rate * error * xi for w, xi in zip(weights, x)]
    new_bias = bias + learning_rate * error
    
    return {
        "score": score,
        "initial_prediction": initial_prediction,
        "error": error,
        "weights": new_weights,
        "bias": new_bias
    }


def independent_perceptron_step(
    weights: List[float],
    bias: float,
    x: List[float],
    target: int,
    learning_rate: float = 1.0
) -> Dict[str, Any]:
    """
    Independent functional reference calculation using explicit vector logic.
    """
    dot_product = weights[0] * x[0] + weights[1] * x[1] + bias
    pred = 0 if dot_product <= 0 else 1
    err = target - pred
    w0 = weights[0] + learning_rate * err * x[0]
    w1 = weights[1] + learning_rate * err * x[1]
    b = bias + learning_rate * err
    return {
        "score": dot_product,
        "initial_prediction": pred,
        "error": err,
        "weights": [w0, w1],
        "bias": b
    }


def compare_approaches(
    weights: List[float],
    bias: float,
    x: List[float],
    target: int,
    learning_rate: float = 1.0
) -> Dict[str, Any]:
    """
    Compares direct and independent implementations for equivalence.
    """
    direct_res = direct_perceptron_step(weights, bias, x, target, learning_rate)
    ref_res = independent_perceptron_step(weights, bias, x, target, learning_rate)
    
    match = (
        direct_res["initial_prediction"] == ref_res["initial_prediction"] and
        direct_res["weights"] == ref_res["weights"] and
        direct_res["bias"] == ref_res["bias"]
    )
    return {
        "direct": direct_res,
        "reference": ref_res,
        "equivalent": match
    }


if __name__ == "__main__":
    w = [0.0, 0.0]
    b = 0.0
    x = [1.0, 2.0]
    target = 1
    comparison = compare_approaches(w, b, x, target, 1.0)
    print("Comparison Result:", comparison)
