"""
Module that transforms dataset
"""

from collections.abc import Mapping

from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase


def prepare_batch(
    batch: Mapping[str, list], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> dict[str, list]:
    """
    Method that prepares single batch of data for the model

    Args:
      batch (Mapping[str, list]): Single batch of data
      tokenizer (PreTrainedTokenizerBase): Model's tokenizer
      max_length (int): Max input length

    Returns:
      dict[str, list]: Preprocessed batched data
    """
    examples: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}

    for question, label in zip(batch["question"], batch["label"]):
        prefix = tokenizer(
            f"Problem:\n{question['problem']}\n\nSolution:\n",
            add_special_tokens=False,
        )["input_ids"]

        for step in label["steps"]:
            for completion in step["completions"]:
                if completion["rating"] is None or completion["flagged"]:
                    continue

                tokens = tokenizer(
                    completion["text"] + "\n\n", add_special_tokens=False
                )["input_ids"]
                input_ids = prefix + tokens

                if len(input_ids) > max_length:
                    continue

                examples["input_ids"].append(input_ids)
                examples["attention_mask"].append([1] * len(input_ids))
                examples["labels"].append(completion["rating"] + 1)

            chosen = step["chosen_completion"]
            text = (
                step["human_completion"]
                if chosen is None
                else step["completions"][chosen]["text"]
            )
            if text is None:
                break

            prefix += tokenizer(text + "\n\n", add_special_tokens=False)["input_ids"]
            if len(prefix) >= max_length:
                break

    return examples


def prepare_dataset(
    dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> DatasetDict:
    """
    Method that preprocesses whole dataset for the training

    Args:
      dataset (DatasetDict): Full dataset
      tokenizer (PreTrainedTokenizerBase): Model's tokenizer
      max_length (int): Max input length

    Returns:
      DatasetDict: Preprocessed dataset
    """
    return dataset.map(
        prepare_batch,
        batched=True,
        batch_size=64,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=dataset["train"].column_names,
    )
