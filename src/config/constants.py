"""
Module that contains project constants
"""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw"
OUTPUT_PATH = ROOT_DIR / "outputs" / "baseline"

DATASET_NAME = "tasksource/PRM800K"
MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"
MAX_LENGTH = 512
SEED = 3407
