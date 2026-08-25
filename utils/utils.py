from utils.parser import extract_answer
from utils.grader import math_equal

DATASET_KEYS = {
    "openai/gsm8k": {"question": "question", "answer": "answer"},
    "hendrycks/competition_math": {"question": "problem", "answer": "solution"},
    "datasets/converted_aime_dataset": {"question": "problem", "answer": "solution"},
    "di-zhang-fdu/MATH500": {"question": "problem", "answer": "solution"},
    "datasets/compression_dataset": {"question": "problem", "answer": "solution"},
    "hiyouga/math12k": {"question": "problem", "answer": "answer"},
}

RESPONSE_EXTRACTOR = {
    "openai/gsm8k": lambda x: extract_answer(x, data_name="gsm8k"),
    "hendrycks/competition_math": lambda x: extract_answer(x, data_name="math"),
    "di-zhang-fdu/MATH500": lambda x: extract_answer(x, data_name="math"),
    "datasets/compression_dataset": lambda x: extract_answer(x, data_name="math"),
    "datasets/converted_aime_dataset": lambda x: extract_answer(x, data_name="math"),
    "hiyouga/math12k": lambda x: extract_answer(x, data_name="math"),
}

# Existing datasets store worked solutions, so extract their final answer before
# passing it to Math-Verify. Math12K already stores a bare answer and must not use
# extract_answer's last-number fallback.
REFERENCE_EXTRACTOR = {
    "openai/gsm8k": lambda x: extract_answer(x, data_name="gsm8k"),
    "hendrycks/competition_math": lambda x: extract_answer(x, data_name="math"),
    "di-zhang-fdu/MATH500": lambda x: extract_answer(x, data_name="math"),
    "datasets/compression_dataset": lambda x: extract_answer(x, data_name="math"),
    "datasets/converted_aime_dataset": lambda x: extract_answer(x, data_name="math"),
    "hiyouga/math12k": lambda x: str(x).strip(),
}

RESPONSE_COMPARATOR = {
    "openai/gsm8k": lambda x, y: math_equal(x, y, timeout=True),
    "hendrycks/competition_math": lambda x, y: math_equal(x, y, timeout=True),
    "di-zhang-fdu/MATH500": lambda x, y: math_equal(x, y, timeout=True),
    "datasets/compression_dataset": lambda x, y: math_equal(x, y, timeout=True),
    "datasets/converted_aime_dataset": lambda x, y: math_equal(x, y, timeout=True),
    "hiyouga/math12k": lambda x, y: math_equal(x, y, timeout=True),
}
