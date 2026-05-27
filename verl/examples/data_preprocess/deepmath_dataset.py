# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess DeepMath-103K (local parquet shards or Hugging Face) to verl parquet format.

Schema follows `zwhe99/DeepMath-103K`: `question`, `final_answer`, optional `difficulty`,
`topic`, `r1_solution_*`. Only a train split exists upstream; this script uses
``train_test_split`` to build a held-out ``test`` split for ``test.parquet``.
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(answer_str: str) -> str:
    """Ground truth for rule-based math reward: prefer \\boxed{} if present, else raw final answer."""
    if answer_str is None:
        return ""
    text = answer_str.strip()
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        return remove_boxed(boxed)
    return text


def load_raw_dataset(local_dataset_path: str | None) -> datasets.Dataset:
    data_source = "zwhe99/DeepMath-103K"
    if local_dataset_path is None:
        print(f"Loading {data_source} from Hugging Face...", flush=True)
        ddict = datasets.load_dataset(data_source)
    else:
        path = os.path.expanduser(local_dataset_path)
        print(f"Loading DeepMath parquet from {path}...", flush=True)
        if os.path.isdir(path):
            ddict = datasets.load_dataset("parquet", data_dir=path)
        elif os.path.isfile(path):
            ddict = datasets.DatasetDict({"train": datasets.load_dataset("parquet", data_files=path, split="train")})
        else:
            raise FileNotFoundError(f"Not a file or directory: {path}")

    if "train" not in ddict:
        raise KeyError(f"Expected a 'train' split in dataset, got keys: {list(ddict.keys())}")
    return ddict["train"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Directory of train-*.parquet shards, a single .parquet file, or omit to load from Hugging Face.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/deepmath",
        help="Directory to write train.parquet / test.parquet.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.01,
        help="Fraction held out as test split (DeepMath release is train-only).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")

    args = parser.parse_args()

    data_source = "zwhe99/DeepMath-103K"
    full_train = load_raw_dataset(args.local_dataset_path)

    split = full_train.train_test_split(test_size=args.test_size, seed=args.seed)
    train_dataset = split["train"]
    test_dataset = split["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split: str):
        def process_fn(example, idx):
            question = example.get("question")
            if question is None and "problem" in example:
                question = example["problem"]
            if question is None:
                raise KeyError("Example has no 'question' (or fallback 'problem') field.")

            final_answer = example.get("final_answer")
            if final_answer is None and "solution" in example:
                final_answer = example["solution"]
            if final_answer is None:
                raise KeyError("Example has no 'final_answer' (or fallback 'solution') field.")

            prompt_text = question.rstrip() + " " + instruction_following
            solution = extract_solution(final_answer)

            extra_info = {"split": split, "index": idx}
            if example.get("topic") is not None:
                extra_info["topic"] = example["topic"]
            if example.get("difficulty") is not None:
                extra_info["difficulty"] = example["difficulty"]

            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": extra_info,
            }

        return process_fn

    train_cols = train_dataset.column_names
    test_cols = test_dataset.column_names
    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        remove_columns=train_cols,
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test"),
        with_indices=True,
        remove_columns=test_cols,
    )

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2)
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(test_dataset[0], f, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
