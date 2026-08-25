"""Serve outcome and length-aware rewards to the OpenRLHF trainer."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from flask import Flask, jsonify, request
from transformers import AutoTokenizer

from utils import DATASET_KEYS, REFERENCE_EXTRACTOR
from utils.math_verifier import verify_math_answer


app = Flask(__name__)

_DATASET_CONFIGS = {
    "openai/gsm8k": "main",
    "hendrycks/competition_math": "main",
}


def load_dataset_dicts(dataset_names):
    """Load each dataset and index its raw gold answers by question."""
    dataset_dict = {}
    print(f"Loading {dataset_names}...")
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_KEYS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        if os.path.isdir(dataset_name):
            dataset = load_from_disk(dataset_name)
        else:
            config_name = _DATASET_CONFIGS.get(dataset_name)
            dataset = load_dataset(dataset_name, config_name) if config_name else load_dataset(dataset_name)

        if isinstance(dataset, DatasetDict):
            if "train" not in dataset:
                raise ValueError(f"Dataset {dataset_name} does not contain a train split")
            dataset = dataset["train"]

        print(f"Picking {dataset_name} consisting of {len(dataset)} examples.")
        question_key = DATASET_KEYS[dataset_name]["question"]
        answer_key = DATASET_KEYS[dataset_name]["answer"]
        dataset_dict[dataset_name] = {entry[question_key]: entry[answer_key] for entry in dataset}

    return dataset_dict


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _response_info(response, tokenizer):
    """Return decoded-token length and whether the response contains EOS."""
    token_ids = tokenizer.encode(response, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    response_len = len(tokenizer.encode(decoded, add_special_tokens=False))
    return response_len, tokenizer.eos_token_id in token_ids


def _verify_pairs(pairs):
    """Score unique response/reference pairs, using persistent worker processes when configured."""
    verifier_pool = app.config.get("verifier_pool")
    if verifier_pool is None:
        return [verify_math_answer(response, reference) for response, reference in pairs]

    futures = [verifier_pool.submit(verify_math_answer, response, reference) for response, reference in pairs]
    scores = []
    for future in futures:
        try:
            scores.append(float(future.result()))
        except Exception as exc:
            app.logger.warning("Math verification failed: %s", exc)
            scores.append(0.0)
    return scores


def _warm_verifier_pool(verifier_pool, worker_count):
    """Start verifier workers before Flask creates request-handling threads."""
    futures = [verifier_pool.submit(verify_math_answer, r"\boxed{0}", "0") for _ in range(worker_count)]
    if any(future.result() != 1.0 for future in futures):
        raise RuntimeError("Math-Verify worker failed its startup check")


@app.route("/query", methods=["POST"])
def query():
    """Compute binary correctness and the configured length-aware reward."""
    query_dict = None
    try:
        metrics = {"rewards": []}
        for dataset_name in app.config["dataset_names"]:
            metrics[f"{dataset_name}_accuracy"] = []
            metrics[f"{dataset_name}_response_length"] = []
            metrics[f"is_{dataset_name}"] = []

        tokenizer = app.config["tokenizer"]
        query_dict = request.get_json()
        query_items = query_dict.get("query", [])

        contexts = []
        response_info = {}
        candidates = {}
        accuracy_by_key = {}

        for query_item in query_items:
            aux_info = query_item.get("aux_info") or {}
            dataset_name = aux_info.get("dataset_name")
            if dataset_name not in DATASET_KEYS:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            question_key = DATASET_KEYS[dataset_name]["question"]
            question = aux_info[question_key]
            answer = app.config["dataset_dict"][dataset_name].get(question)
            if answer is None:
                raise KeyError(f"No answer found for question in {dataset_name}: {question[:120]}")
            reference = REFERENCE_EXTRACTOR[dataset_name](answer)

            response = query_item.get("response")
            if response is None:
                raise ValueError("Query item is missing a response")
            all_responses = aux_info.get("all_responses") or [response]
            contexts.append(
                {
                    "dataset_name": dataset_name,
                    "question": question,
                    "response": response,
                    "all_responses": all_responses,
                }
            )

            for candidate_response in [response, *all_responses]:
                if candidate_response not in response_info:
                    response_info[candidate_response] = _response_info(candidate_response, tokenizer)

                cache_key = (dataset_name, question, candidate_response)
                _, contains_eos = response_info[candidate_response]
                if app.config["check_eos"] and not contains_eos:
                    accuracy_by_key[cache_key] = 0.0
                elif cache_key not in candidates:
                    candidates[cache_key] = (candidate_response, reference)

        candidate_keys = list(candidates)
        candidate_scores = _verify_pairs([candidates[key] for key in candidate_keys])
        accuracy_by_key.update(dict(zip(candidate_keys, candidate_scores)))

        correct_lengths = {}
        visited_groups = set()
        for context in contexts:
            group_key = (context["dataset_name"], context["question"])
            if group_key in visited_groups:
                continue
            visited_groups.add(group_key)

            lengths = []
            for response in context["all_responses"]:
                cache_key = (*group_key, response)
                if accuracy_by_key[cache_key] > 0:
                    lengths.append(response_info[response][0])
            correct_lengths[group_key] = lengths

        for context in contexts:
            dataset_name = context["dataset_name"]
            group_key = (dataset_name, context["question"])
            response = context["response"]
            response_len = response_info[response][0]
            accuracy = accuracy_by_key[(*group_key, response)]

            if app.config["reward_type"] == "sigmoid":
                if accuracy > 0:
                    lengths = correct_lengths[group_key] or [response_len]
                    relative_length = (response_len - np.mean(lengths)) / (np.std(lengths) + 1e-7)
                    reward = accuracy * (1 - app.config["alpha"] * sigmoid(relative_length))
                else:
                    reward = 0.0
            elif app.config["reward_type"] == "linear":
                reward = accuracy
            else:
                raise ValueError(f"Unsupported reward type: {app.config['reward_type']}")

            metrics["rewards"].append(float(reward))
            for metric_dataset_name in app.config["dataset_names"]:
                if metric_dataset_name == dataset_name:
                    metrics[f"is_{metric_dataset_name}"].append(1.0)
                    metrics[f"{metric_dataset_name}_accuracy"].append(float(accuracy))
                    metrics[f"{metric_dataset_name}_response_length"].append(response_len)
                else:
                    metrics[f"is_{metric_dataset_name}"].append(float("nan"))
                    metrics[f"{metric_dataset_name}_accuracy"].append(float("nan"))
                    metrics[f"{metric_dataset_name}_response_length"].append(float("nan"))

        return jsonify(metrics), 200

    except Exception as exc:
        if query_dict is not None:
            with open("error.json", "w", encoding="utf-8") as error_file:
                json.dump(query_dict, error_file, indent=4)
        app.logger.exception("Reward query failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, default="0.0.0.0:100")
    parser.add_argument("--dataset_names", "--dataset", dest="dataset_names", default="openai/gsm8k")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--reward_type", choices=["linear", "sigmoid"], default="linear")
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--check_eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verifier_workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()

    dataset_names = [name.strip() for name in args.dataset_names.split(",")]
    app.config["dataset_names"] = dataset_names
    app.config["dataset_dict"] = load_dataset_dicts(dataset_names)
    app.config["tokenizer"] = AutoTokenizer.from_pretrained(args.tokenizer)
    app.config["reward_type"] = args.reward_type
    app.config["alpha"] = args.alpha
    app.config["check_eos"] = args.check_eos

    print(f"Server will start at http://{args.address}/query")
    verifier_workers = max(1, args.verifier_workers)
    with ProcessPoolExecutor(max_workers=verifier_workers) as verifier_pool:
        _warm_verifier_pool(verifier_pool, verifier_workers)
        app.config["verifier_pool"] = verifier_pool
        app.run(host=args.address.split(":")[0], port=int(args.address.split(":")[1]))
