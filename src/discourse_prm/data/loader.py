"""
Module that loads and filters dataset
"""

from random import Random

from datasets import DatasetDict, load_dataset

from config.constants import RAW_DATA_PATH, SEED


def obtain_dataset(dataset_id: str) -> DatasetDict:
    """
    Method that loads dataset and filters it

    Args:
      dataset_id (str): Dataset name on the HF platform

    Returns:
      DatasetDict: Filtered dataset with 3 splits
    """
    dataset = load_dataset(
        dataset_id,
        data_files={"train": "phase2_train.jsonl", "test": "phase2_test.jsonl"},
        cache_dir=str(RAW_DATA_PATH),
    )
    dataset = dataset.filter(
        lambda row: (
            not row["is_quality_control_question"]
            and not row["is_initial_screening_question"]
            and row["label"]["finish_reason"] in ("solution", "found_error")
        )
    )

    problems = sorted({row["question"]["problem"] for row in dataset["train"]})
    Random(SEED).shuffle(problems)
    validation_problems = set(problems[: len(problems) // 20])

    return DatasetDict(
        train=dataset["train"].filter(
            lambda row: row["question"]["problem"] not in validation_problems
        ),
        validation=dataset["train"].filter(
            lambda row: row["question"]["problem"] in validation_problems
        ),
        test=dataset["test"],
    )
