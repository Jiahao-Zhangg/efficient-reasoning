from openrlhf.datasets.prompts_dataset import preprocess_data


def test_chat_template_omits_empty_system_role_and_formats_user_prompt():
    rendered_messages = None

    def render(messages, tokenize, add_generation_prompt):
        nonlocal rendered_messages
        rendered_messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered prompt"

    data = {"dataset_name": "hiyouga/math12k", "problem": "What is 2 + 2?"}
    prompt, aux_info = preprocess_data(
        data,
        input_template="{}\nShow your work.",
        apply_chat_template=render,
    )

    assert prompt == "rendered prompt"
    assert rendered_messages == [{"role": "user", "content": "What is 2 + 2?\nShow your work."}]
    assert aux_info is data


def test_chat_template_includes_explicit_system_prompt():
    rendered_messages = None

    def render(messages, **kwargs):
        nonlocal rendered_messages
        rendered_messages = messages
        return "rendered prompt"

    data = {"dataset_name": "hiyouga/math12k", "problem": "What is 2 + 2?"}
    preprocess_data(data, apply_chat_template=render, system_prompt="You are a math tutor.")

    assert rendered_messages == [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]


def test_compression_prompt_preserves_sample_gold_without_showing_it_to_model():
    data = {
        "dataset_name": "datasets/compression_dataset",
        "problem": "How much is 2 + 2?",
        "solution": "A worked solution ending in 4",
        "extracted": "4",
    }
    prompt, aux_info = preprocess_data(data, input_template="Question: {}")

    assert prompt == "Question: How much is 2 + 2?"
    assert aux_info is data
    assert aux_info["extracted"] == "4"
