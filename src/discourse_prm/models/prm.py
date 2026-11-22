"""
Module that creates wrapper for the model that adds extra heads to the backbone
and controls gradient flow.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3_5TextModel


class PRM(nn.Module):
    """
    Wrapper for the model that adds extra heads to the backbone
    """

    base_model_prefix = "backbone"

    def __init__(self, backbone: Qwen3_5TextModel) -> None:
        """
        Initialize the class instance

        Args:
          backbone (Qwen3_5TextModel): Base model with trimmed head and visual block
        """
        super().__init__()
        self.config = backbone.config
        self.config.num_labels = 3
        self.name_or_path = backbone.name_or_path
        self.backbone = backbone
        self.reward_head = nn.Linear(backbone.config.hidden_size, 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Overriden forward method

        Args:
          input_ids (torch.Tensor): Input tokens ids
          attention_mask (torch.Tensor): Attention mask for the generation
          labels (None): Labels for loss and extra metrics calculation

        Returns:
          dict[str, torch.Tensor | None]: Loss and raw logits
        """
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        positions = attention_mask.sum(dim=1) - 1
        batch = torch.arange(input_ids.shape[0], device=input_ids.device)
        logits = self.reward_head(hidden[batch, positions].float())
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}
