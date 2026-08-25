"""Math-Verify scoring compatible with MaxRL's outcome grader."""

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


_VERIFY_FUNC = None


def _get_verify_func():
    """Create one verifier per process, matching MaxRL's worker-local setup."""
    global _VERIFY_FUNC
    if _VERIFY_FUNC is None:
        _VERIFY_FUNC = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    return _VERIFY_FUNC


def verify_math_answer(model_output, ground_truth, timeout_score=0.0):
    """Return a binary Math-Verify score for a model response and bare gold answer.

    Math-Verify expects the reference to be in a LaTeX environment. Wrapping the
    already-normalized gold prevents bare fractions or symbolic expressions from
    being reduced to their last numeric token.
    """
    if model_output is None or ground_truth is None:
        return float(timeout_score)

    model_output = str(model_output).strip()
    ground_truth = str(ground_truth).strip()
    if not model_output or not ground_truth:
        return float(timeout_score)

    ground_truth_boxed = f"\\boxed{{{ground_truth}}}"
    try:
        score, _ = _get_verify_func()([ground_truth_boxed], [model_output])
        return float(score)
    except TimeoutException:
        return float(timeout_score)
    except Exception:
        return 0.0
