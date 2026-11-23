"""
Applied Lab R38-T008: Tensor-Shape Contract & Failure Recovery Preprocessing
Lesson: lessons/4-ComputerVision/09-Autoencoders

Contract:
- Preprocess synthetic image batches of shape (N, H, W) with numeric pixels in [0, 255].
- Normalize to [0.0, 1.0] while preserving identical batch shape.
- Canonical fixture: [[[0, 255], [128, 64]]] stays shape (1, 2, 2), 0 -> 0.0, 255 -> 1.0.
- Pixels < 0 or > 255 or invalid shapes raise ValueError / TypeError.
- Focus: Exercise a documented invalid/missing-data case, record the diagnostic error,
  and demonstrate state recovery on the immediately following valid input without
  preprocessor state corruption.
"""

from typing import List, Tuple, Dict, Any, Union, Optional

Numeric = Union[int, float]
Image = List[List[Numeric]]
Batch = List[Image]


class AutoencoderBatchPreprocessor:
    """
    Validates and normalizes image batches for autoencoder reconstruction pipelines.
    Tracks execution history and supports graceful error reporting and recovery.
    """
    def __init__(self):
        self.processed_count: int = 0
        self.error_log: List[Dict[str, Any]] = []

    def get_shape(self, batch: Any) -> Tuple[int, int, int]:
        if not isinstance(batch, list) or len(batch) == 0:
            raise ValueError("Batch must be a non-empty list.")
        n = len(batch)
        if not isinstance(batch[0], list) or len(batch[0]) == 0:
            raise ValueError("Batch elements must be non-empty 2D images.")
        h = len(batch[0])
        if not isinstance(batch[0][0], list) or len(batch[0][0]) == 0:
            raise ValueError("Image rows must be non-empty lists.")
        w = len(batch[0][0])

        for bi, img in enumerate(batch):
            if not isinstance(img, list) or len(img) != h:
                raise ValueError(f"Inconsistent height at image {bi}: expected {h}")
            for ri, row in enumerate(img):
                if not isinstance(row, list) or len(row) != w:
                    raise ValueError(f"Inconsistent width at image {bi}, row {ri}: expected {w}")
        return (n, h, w)

    def normalize(self, batch: Batch) -> List[List[List[float]]]:
        """
        Validates pixel ranges and normalizes batch to [0.0, 1.0].
        Raises ValueError or TypeError on invalid input.
        """
        n, h, w = self.get_shape(batch)
        normalized: List[List[List[float]]] = []

        for bi, img in enumerate(batch):
            norm_img: List[List[float]] = []
            for ri, row in enumerate(img):
                norm_row: List[float] = []
                for ci, pixel in enumerate(row):
                    if pixel is None:
                        raise ValueError(f"Missing pixel data at ({bi}, {ri}, {ci})")
                    if not isinstance(pixel, (int, float)) or isinstance(pixel, bool):
                        raise TypeError(f"Pixel at ({bi}, {ri}, {ci}) must be numeric, got {type(pixel).__name__}")
                    if pixel < 0 or pixel > 255:
                        raise ValueError(f"Pixel value {pixel} at ({bi}, {ri}, {ci}) out of valid range [0, 255]")
                    norm_row.append(float(pixel) / 255.0)
                norm_img.append(norm_row)
            normalized.append(norm_img)

        self.processed_count += 1
        return normalized

    def safe_process_with_recovery(self, batch: Any) -> Dict[str, Any]:
        """
        Processes batch, catching and logging errors, demonstrating preprocessor resilience.
        """
        try:
            result = self.normalize(batch)
            return {
                "success": True,
                "data": result,
                "shape": (len(result), len(result[0]), len(result[0][0])),
                "error": None
            }
        except (ValueError, TypeError) as err:
            err_record = {
                "success": False,
                "data": None,
                "shape": None,
                "error": str(err)
            }
            self.error_log.append(err_record)
            return err_record


if __name__ == "__main__":
    processor = AutoencoderBatchPreprocessor()
    valid_batch = [[[0, 255], [128, 64]]]
    invalid_batch = [[[0, -50], [128, 64]]]

    print("1. Process invalid:")
    res_err = processor.safe_process_with_recovery(invalid_batch)
    print(res_err)

    print("2. Recover on valid:")
    res_ok = processor.safe_process_with_recovery(valid_batch)
    print(res_ok)
