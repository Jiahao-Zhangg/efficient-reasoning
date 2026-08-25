from datasets import Dataset

from openrlhf.utils.utils import _filter_empty_math12k_rows


class FakeStrategy:
    def __init__(self):
        self.messages = []

    def print(self, message):
        self.messages.append(message)


def test_math12k_filter_matches_maxrl_preprocessing():
    dataset = Dataset.from_dict(
        {
            "problem": ["What is 2 + 2?", "", "   ", "What is 3 + 3?"],
            "answer": ["4", "5", "   ", "6"],
        }
    )
    strategy = FakeStrategy()

    filtered = _filter_empty_math12k_rows(dataset, "hiyouga/math12k", "train", strategy)

    assert filtered["problem"] == ["What is 2 + 2?", "What is 3 + 3?"]
    assert strategy.messages == ["Skipping 2 train rows with an empty problem or answer"]


def test_other_datasets_are_not_filtered():
    dataset = Dataset.from_dict({"problem": [""], "answer": [""]})
    strategy = FakeStrategy()

    filtered = _filter_empty_math12k_rows(dataset, "some/other-dataset", "train", strategy)

    assert filtered is dataset
    assert strategy.messages == []
