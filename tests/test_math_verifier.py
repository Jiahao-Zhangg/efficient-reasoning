import pytest

from reward_server.math_server import _reference_for_sample, app
from utils import DATASET_KEYS, REFERENCE_EXTRACTOR
from utils.math_verifier import verify_math_answer


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        has_eos = text.endswith("<eos>")
        text = text.removesuffix("<eos>")
        token_ids = [ord(character) + 1 for character in text]
        return token_ids + ([self.eos_token_id] if has_eos else [])

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id - 1) for token_id in token_ids if token_id != self.eos_token_id)


@pytest.mark.parametrize(
    ("model_output", "ground_truth"),
    [
        (r"The answer is \boxed{\frac{7}{20}}.", r"\frac{7}{20}"),
        (r"\boxed{(y+1)^2}", r"y^2+2y+1"),
        (r"\boxed{(-\infty,3)\cup(3,\infty)}", r"(-\infty, 3) \cup (3, \infty)"),
        (r"\boxed{1-\sqrt{2},1+\sqrt{2}}", r"1+\sqrt{2},1-\sqrt{2}"),
        (r"\boxed{C}", r"\text{(C)}"),
        (r"\boxed{even}", r"\text{even}"),
    ],
)
def test_verify_math_answer_supports_math12k_formats(model_output, ground_truth):
    assert verify_math_answer(model_output, ground_truth) == 1.0


def test_verify_math_answer_rejects_wrong_and_empty_answers():
    assert verify_math_answer(r"\boxed{20}", r"\frac{7}{20}") == 0.0
    assert verify_math_answer("", r"\frac{7}{20}") == 0.0
    assert verify_math_answer(r"\boxed{1}", "") == 0.0


def test_math12k_reference_is_not_reduced_to_last_number():
    reference = r"\frac{7}{20}"
    assert REFERENCE_EXTRACTOR["hiyouga/math12k"](reference) == reference


@pytest.mark.parametrize("reference", ["0", "168089", r"\frac{7}{20}", "32x - 46", r"\text{(C)}"])
def test_compression_reference_uses_extracted_without_reparsing(reference):
    dataset_name = "datasets/compression_dataset"
    assert DATASET_KEYS[dataset_name]["answer"] == "extracted"
    assert REFERENCE_EXTRACTOR[dataset_name](reference) == reference
    assert _reference_for_sample(
        dataset_name,
        {"problem": "question", "extracted": reference, "solution": r"Wrong solution: \boxed{99}"},
    ) == reference


@pytest.mark.parametrize("aux_info", [{}, {"extracted": None}, {"extracted": ""}, {"extracted": "  "}])
def test_compression_reference_requires_sample_gold(aux_info):
    with pytest.raises(ValueError, match="non-empty extracted"):
        _reference_for_sample("datasets/compression_dataset", {"problem": "question", **aux_info})


def test_reward_server_keeps_duplicate_questions_with_different_gold_separate(monkeypatch):
    dataset_name = "datasets/compression_dataset"
    question = "A duplicated question with conflicting stored answers"
    responses = [r"\boxed{2}<eos>", r"A longer response: \boxed{3}<eos>"]
    config = {
        "TESTING": True,
        "dataset_names": [dataset_name],
        # No question lookup is available: gold must come from each sample.
        "dataset_dict": {},
        "tokenizer": FakeTokenizer(),
        "reward_type": "sigmoid",
        "alpha": 0.1,
        "check_eos": True,
        "verifier_pool": None,
    }
    for key, value in config.items():
        monkeypatch.setitem(app.config, key, value)
    payload = {
        "query": [
            {
                "response": response,
                "aux_info": {
                    "dataset_name": dataset_name,
                    "problem": question,
                    "extracted": gold,
                    "solution": r"Do not use this: \boxed{99}",
                    "all_responses": responses,
                },
            }
            for gold in ["2", "3"]
            for response in responses
        ]
    }

    response = app.test_client().post("/query", json=payload)
    assert response.status_code == 200
    metrics = response.get_json()
    assert metrics[f"{dataset_name}_accuracy"] == [1.0, 0.0, 0.0, 1.0]
    assert metrics["rewards"] == pytest.approx([0.95, 0.0, 0.0, 0.95])


def test_reward_server_verifies_bare_math12k_gold():
    dataset_name = "hiyouga/math12k"
    question = "What is seven twentieths?"
    correct_response = r"Reasoning... \boxed{\frac{7}{20}}<eos>"
    wrong_response = r"Reasoning... \boxed{20}<eos>"
    all_responses = [correct_response, wrong_response]

    app.config.update(
        TESTING=True,
        dataset_names=[dataset_name],
        dataset_dict={dataset_name: {question: r"\frac{7}{20}"}},
        tokenizer=FakeTokenizer(),
        reward_type="sigmoid",
        alpha=0.1,
        check_eos=True,
        verifier_pool=None,
    )
    payload = {
        "query": [
            {
                "response": response,
                "aux_info": {
                    "dataset_name": dataset_name,
                    "problem": question,
                    "all_responses": all_responses,
                },
            }
            for response in all_responses
        ]
    }

    response = app.test_client().post("/query", json=payload)
    assert response.status_code == 200
    metrics = response.get_json()
    assert metrics[f"{dataset_name}_accuracy"] == [1.0, 0.0]
    assert metrics["rewards"] == pytest.approx([0.95, 0.0])


def test_reward_server_assigns_zero_without_required_eos():
    dataset_name = "hiyouga/math12k"
    question = "What is seven twentieths?"
    response_text = r"\boxed{\frac{7}{20}}"
    app.config.update(
        TESTING=True,
        dataset_names=[dataset_name],
        dataset_dict={dataset_name: {question: r"\frac{7}{20}"}},
        tokenizer=FakeTokenizer(),
        reward_type="linear",
        alpha=0.0,
        check_eos=True,
        verifier_pool=None,
    )
    payload = {
        "query": [
            {
                "response": response_text,
                "aux_info": {
                    "dataset_name": dataset_name,
                    "problem": question,
                    "all_responses": [response_text],
                },
            }
        ]
    }

    response = app.test_client().post("/query", json=payload)
    assert response.status_code == 200
    metrics = response.get_json()
    assert metrics[f"{dataset_name}_accuracy"] == [0.0]
    assert metrics[f"{dataset_name}_response_length"][0] > 0
    assert metrics["rewards"] == [0.0]
