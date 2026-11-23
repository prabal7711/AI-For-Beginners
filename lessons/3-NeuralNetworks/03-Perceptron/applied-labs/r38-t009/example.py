"""
Applied Lab R38-T009: Single Perceptron Update — Controlled Configuration Evaluation
Lesson: lessons/3-NeuralNetworks/03-Perceptron

Contract:
- Implement single perceptron prediction and weight update step on a 2D fixture:
  * Initial weights=[0, 0], bias=0, input=[1, 2], target=1, learning_rate=1.0.
  * Documented zero-score tie rule: score <= 0 -> prediction = 0; score > 0 -> prediction = 1.
  * Initial score = 0*1 + 0*2 + 0 = 0 -> initial prediction = 0.
  * Error = target - prediction = 1 - 0 = 1.
  * Updated weights = [1.0, 2.0], updated bias = 1.0.
- Focus: Controlled experiment comparing three learning rate configurations:
  * Config A: lr = 0.1
  * Config B: lr = 0.5
  * Config C: lr = 1.0
  Report correctness, step magnitude (L2 norm of delta_w), and post-update decision margin.
  Explicitly note: These metrics evaluate single-step update dynamics without claiming real-world
  classification accuracy.
"""

import math
from typing import List, Dict, Any


def perceptron_update_step(
    weights: List[float],
    bias: float,
    x: List[float],
    target: int,
    learning_rate: float = 1.0
) -> Dict[str, Any]:
    """
    Executes a single perceptron prediction and update step.
    """
    if len(weights) != len(x):
        raise ValueError(f"Dimension mismatch: weights length {len(weights)} != input length {len(x)}")
    if target not in (0, 1):
        raise ValueError("Target must be 0 or 1.")

    score = sum(w * xi for w, xi in zip(weights, x)) + bias
    prediction = 1 if score > 0 else 0
    error = target - prediction

    new_weights = [w + learning_rate * error * xi for w, xi in zip(weights, x)]
    new_bias = bias + learning_rate * error

    # Post-update metrics
    delta_w = [nw - ow for nw, ow in zip(new_weights, weights)]
    l2_step_size = math.sqrt(sum(d ** 2 for d in delta_w))
    post_score = sum(nw * xi for nw, xi in zip(new_weights, x)) + new_bias

    return {
        "score": score,
        "prediction": prediction,
        "error": error,
        "weights": new_weights,
        "bias": new_bias,
        "l2_step_size": l2_step_size,
        "post_score": post_score
    }


def run_configuration_experiment(
    weights: List[float],
    bias: float,
    x: List[float],
    target: int,
    learning_rates: List[float] = [0.1, 0.5, 1.0]
) -> Dict[str, Any]:
    """
    Runs controlled evaluation across three configurations and analyzes observed trade-offs.
    """
    results: Dict[str, Any] = {}
    for lr in learning_rates:
        cfg_name = f"lr_{lr}"
        step_res = perceptron_update_step(list(weights), bias, list(x), target, lr)
        results[cfg_name] = {
            "learning_rate": lr,
            "updated_weights": step_res["weights"],
            "updated_bias": step_res["bias"],
            "step_size": step_res["l2_step_size"],
            "post_score": step_res["post_score"]
        }

    trade_off_analysis = (
        "Trade-off Summary: Increasing learning rate from 0.1 to 1.0 scales weight displacement "
        "and post-update score linearly. Higher learning rate produces immediate correction margin "
        "on the current sample but risks overshooting and oscillation across multi-sample batches."
    )

    return {
        "configurations": results,
        "trade_off_analysis": trade_off_analysis
    }


if __name__ == "__main__":
    w0 = [0.0, 0.0]
    b0 = 0.0
    x0 = [1.0, 2.0]
    target0 = 1
    exp = run_configuration_experiment(w0, b0, x0, target0)
    print("Experiment Results:", exp)
