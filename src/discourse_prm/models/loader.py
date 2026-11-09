"""
Module that loads model with its tokenizer and adds LoRA matrices to it
"""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
)

from discourse_prm.models.prm import PRM


def obtain_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """
    Method that loads tokenizer and assigns pad and eos tokens from vocab

    Args:
      model_id (str): Model name on the HF platform

    Returns:
      PreTrainedTokenizerBase: Tokenizer for the model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def obtain_model(model_id: str) -> PeftModel:
    """
    Method that loads model, adds extra heads to it and wraps it into the PEFT format

    Args:
      model_id (str): Model name on the HF platform

    Returns:
      PeftModel: Model with LoRA matrices of defined rank added to the backbone and extra heads
    """
    backbone = Qwen3_5TextModel.from_pretrained(
        model_id,
        config=Qwen3_5TextConfig.from_pretrained(model_id),
        dtype=torch.float32,
    )
    backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return get_peft_model(
        PRM(backbone),
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            modules_to_save=["reward_head"],
        ),
    )
