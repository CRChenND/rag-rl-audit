import numpy as np
import torch


def normalize_reward_postprocess_cfg(cfg: dict | None) -> dict:
    raw = cfg or {}
    normalized = {
        "enabled": bool(raw.get("enabled", True)),
        "temperature": float(raw.get("temperature", 1.0)),
        "normalize": str(raw.get("normalize", "none")).lower(),
        "apply_tanh": bool(raw.get("apply_tanh", True)),
        "clip_min": raw.get("clip_min", None),
        "clip_max": raw.get("clip_max", None),
        "eps": float(raw.get("eps", 1e-6)),
        "running_momentum": float(raw.get("running_momentum", 0.95)),
        "update_stats_in_eval": bool(raw.get("update_stats_in_eval", False)),
        "min_count_for_running": int(raw.get("min_count_for_running", 64)),
        "length_penalty": float(raw.get("length_penalty", 0.0)),
        "length_penalty_mode": str(raw.get("length_penalty_mode", "response_tokens")).lower(),
        "length_penalty_scale": str(raw.get("length_penalty_scale", "none")).lower(),
    }

    if normalized["temperature"] <= 0:
        raise ValueError("reward_postprocess.temperature must be > 0")
    if normalized["normalize"] not in {"none", "batch_zscore", "running_zscore"}:
        raise ValueError("reward_postprocess.normalize must be one of: none, batch_zscore, running_zscore")
    if not 0.0 <= normalized["running_momentum"] < 1.0:
        raise ValueError("reward_postprocess.running_momentum must be in [0, 1)")
    if normalized["min_count_for_running"] < 1:
        raise ValueError("reward_postprocess.min_count_for_running must be >= 1")
    if normalized["length_penalty_mode"] not in {"response_tokens", "all_tokens"}:
        raise ValueError("reward_postprocess.length_penalty_mode must be one of: response_tokens, all_tokens")
    if normalized["length_penalty_scale"] not in {"none", "sqrt", "log"}:
        raise ValueError("reward_postprocess.length_penalty_scale must be one of: none, sqrt, log")
    return normalized


def _scale_lengths_torch(lengths: torch.Tensor, scale: str) -> torch.Tensor:
    if scale == "sqrt":
        return torch.sqrt(lengths.clamp_min(1.0))
    if scale == "log":
        return torch.log1p(lengths.clamp_min(0.0))
    return lengths


def apply_squash_clip_torch(values: torch.Tensor, cfg: dict) -> torch.Tensor:
    out = values
    if cfg["apply_tanh"]:
        out = torch.tanh(out)
    if cfg["clip_min"] is not None or cfg["clip_max"] is not None:
        cmin = float("-inf") if cfg["clip_min"] is None else float(cfg["clip_min"])
        cmax = float("inf") if cfg["clip_max"] is None else float(cfg["clip_max"])
        out = torch.clamp(out, min=cmin, max=cmax)
    return out


def apply_reward_postprocess_numpy(scores: np.ndarray, cfg: dict) -> np.ndarray:
    norm_cfg = normalize_reward_postprocess_cfg(cfg)
    if not norm_cfg["enabled"]:
        return scores

    out = scores / norm_cfg["temperature"]
    normalize = norm_cfg["normalize"]
    if normalize in {"batch_zscore", "running_zscore"}:
        # Diagnostics are computed on finite arrays, so running_zscore falls back
        # to array-level z-score for deterministic offline comparison.
        std = float(np.std(out))
        out = (out - float(np.mean(out))) / (std + norm_cfg["eps"])

    if norm_cfg["apply_tanh"]:
        out = np.tanh(out)
    if norm_cfg["clip_min"] is not None or norm_cfg["clip_max"] is not None:
        cmin = -np.inf if norm_cfg["clip_min"] is None else float(norm_cfg["clip_min"])
        cmax = np.inf if norm_cfg["clip_max"] is None else float(norm_cfg["clip_max"])
        out = np.clip(out, a_min=cmin, a_max=cmax)
    return out


def apply_length_penalty_torch(logits: torch.Tensor, lengths: torch.Tensor | None, cfg: dict) -> torch.Tensor:
    norm_cfg = normalize_reward_postprocess_cfg(cfg)
    if lengths is None or norm_cfg["length_penalty"] == 0.0:
        return logits
    scaled = _scale_lengths_torch(lengths, norm_cfg["length_penalty_scale"])
    return logits - norm_cfg["length_penalty"] * scaled
