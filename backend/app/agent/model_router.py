"""Deterministic request-scope model selection for Agent providers."""
from __future__ import annotations
from dataclasses import dataclass


class ModelSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class SelectedModel:
    tier: str
    model_id: str
    policy: str
    reason: str


class ModelRouter:
    def __init__(self, flash_model: str | None, pro_model: str | None, legacy_model: str | None = None, policy: str = "flash_first"):
        self.flash_model = flash_model
        self.pro_model = pro_model
        self.legacy_model = legacy_model
        self.policy = policy

    def select_model(self, intent: str | None, requested_output: str | None, quality_mode: str | None = None, runtime_mode: str | None = None) -> SelectedModel:
        if self.policy != "flash_first":
            raise ModelSelectionError(f"Unsupported agent model policy: {self.policy}")
        if quality_mode not in {None, "economy", "balanced", "high"}:
            raise ModelSelectionError(f"Unsupported quality mode: {quality_mode}")
        if quality_mode == "high":
            if not self.pro_model:
                raise ModelSelectionError("Pro model is not configured for quality_mode=high")
            return SelectedModel("pro", self.pro_model, self.policy, "quality_mode_high")
        model_id = self.flash_model or self.legacy_model
        if not model_id:
            raise ModelSelectionError("Flash model is not configured")
        reason = "flash_first" if self.flash_model else "legacy_model_fallback"
        return SelectedModel("flash", model_id, self.policy, reason)
