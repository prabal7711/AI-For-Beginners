"""
Applied Lab R38-T007: Training-Only Imputation — Fit/Transform & Leakage Prevention
Lesson: lessons/4-ComputerVision/08-TransferLearning

Contract:
- Fit a missing-value mean exclusively on training data, then apply that learned parameter
  to validation observations without refitting (preventing data leakage).
- Training fixture: [2, None, 4] -> non-missing mean is (2 + 4) / 2 = 3.0.
- Validation fixture: [100, None] -> transformed output is [100.0, 3.0] (NOT [100.0, 100.0]).
- All-missing training data (e.g., [None, None]) is rejected with ValueError.
- Worked explanation:
  Data leakage occurs when information from test/validation sets leaks into the training pipeline.
  In transfer learning feature preprocessing or tabular normalization, statistics (mean, std) must
  always be estimated strictly on the training partition and frozen before applying to validation.
"""

from typing import List, Optional, Union, Any

DataPoint = Optional[Union[int, float]]


class MeanImputer:
    """
    Fits mean on training observations and imputes missing values (None)
    across arbitrary target splits without refitting.
    """
    def __init__(self):
        self.fitted_mean: Optional[float] = None

    def fit(self, train_data: List[DataPoint]) -> "MeanImputer":
        if not isinstance(train_data, list):
            raise TypeError(f"train_data must be a list, got {type(train_data).__name__}")
        
        valid_vals = [
            float(x) for x in train_data
            if x is not None and not isinstance(x, bool) and isinstance(x, (int, float))
        ]
        
        if len(valid_vals) == 0:
            raise ValueError("All-missing or empty training input rejected: cannot compute mean.")
        
        self.fitted_mean = sum(valid_vals) / len(valid_vals)
        return self

    def transform(self, data: List[DataPoint]) -> List[float]:
        if self.fitted_mean is None:
            raise RuntimeError("MeanImputer must be fitted on training data before calling transform().")
        if not isinstance(data, list):
            raise TypeError(f"Data must be a list, got {type(data).__name__}")
        
        imputed: List[float] = []
        for i, val in enumerate(data):
            if val is None:
                imputed.append(self.fitted_mean)
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                imputed.append(float(val))
            else:
                raise TypeError(f"Invalid observation at index {i}: {val}")
        return imputed

    def fit_transform(self, train_data: List[DataPoint]) -> List[float]:
        return self.fit(train_data).transform(train_data)


def explain_worked_example():
    """Generates detailed textual breakdown of the worked example."""
    train_data = [2, None, 4]
    val_data = [100, None]

    imputer = MeanImputer()
    imputer.fit(train_data)
    imputed_train = imputer.transform(train_data)
    imputed_val = imputer.transform(val_data)

    explanation = (
        f"1. Training Input: {train_data}\n"
        f"   - Valid observations: 2 and 4.\n"
        f"   - Computed training mean: (2 + 4) / 2 = {imputer.fitted_mean}\n"
        f"   - Imputed training set: {imputed_train}\n"
        f"2. Validation Input: {val_data}\n"
        f"   - Validation non-missing value: 100.\n"
        f"   - If validation data was used to fit, mean would be 100.0 (Data Leakage!).\n"
        f"   - Correctly using stored training mean ({imputer.fitted_mean}) yields: {imputed_val}\n"
    )
    return explanation


if __name__ == "__main__":
    print(explain_worked_example())
