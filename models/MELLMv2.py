"""ME-LLM model for long-horizon time-series forecasting.

The model creates one numerical token and one vocabulary-conditioned semantic
token for each time-series patch. The paired tokens are interleaved, prefixed
with a structured prompt, and processed by a frozen BERT-base encoder. Only the
patch embedding, prototype mapper, cross-attention module, and prediction head
are optimized.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertConfig, BertModel, BertTokenizer

from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


DATASET_DESCRIPTIONS: Dict[str, str] = {
    "etth1": (
        "ETTh1 contains seven hourly electricity-transformer variables, "
        "including oil temperature and load measurements."
    ),
    "etth2": (
        "ETTh2 contains seven hourly electricity-transformer variables, "
        "including oil temperature and load measurements."
    ),
    "ettm1": (
        "ETTm1 contains seven electricity-transformer variables sampled every "
        "15 minutes, including oil temperature and load measurements."
    ),
    "ettm2": (
        "ETTm2 contains seven electricity-transformer variables sampled every "
        "15 minutes, including oil temperature and load measurements."
    ),
    "weather": "Weather contains 21 meteorological variables sampled every 10 minutes.",
    "traffic": "Traffic contains 862 hourly traffic variables.",
}

SUPPORTED_FORECAST_HORIZONS = (96, 192, 336, 720)


def _config_value(configs, name: str, default=None):
    """Read a value from a namespace or mapping."""

    if isinstance(configs, dict):
        return configs.get(name, default)
    return getattr(configs, name, default)


def _normalize_dataset_key(value: object) -> str:
    text = str(value or "").lower().replace("_", "").replace("-", "")
    for key in DATASET_DESCRIPTIONS:
        if key in text:
            return key
    return text


def _resolve_dataset_description(configs) -> Tuple[str, str]:
    """Return the dataset key and its fixed prompt description."""

    explicit = _config_value(configs, "dataset_description", None)
    candidates = (
        _config_value(configs, "data", None),
        _config_value(configs, "dataset", None),
        _config_value(configs, "model_id", None),
        _config_value(configs, "data_path", None),
        _config_value(configs, "root_path", None),
    )

    dataset_key = ""
    for candidate in candidates:
        candidate_key = _normalize_dataset_key(candidate)
        if candidate_key in DATASET_DESCRIPTIONS:
            dataset_key = candidate_key
            break

    if explicit:
        return dataset_key or "custom", str(explicit).strip()
    if dataset_key:
        return dataset_key, DATASET_DESCRIPTIONS[dataset_key]

    raise ValueError(
        "No fixed dataset description is defined. Set dataset_description in the run configuration."
    )


def _format_number(value: float, digits: int = 4) -> str:
    """Format a finite scalar deterministically for the prompt."""

    if not math.isfinite(value):
        raise ValueError("Prompt statistics must be finite.")
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _format_lag_list(lags: Sequence[int]) -> str:
    values = [str(int(lag)) for lag in lags]
    if not values:
        raise ValueError("At least one positive lag is required.")
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + ", and " + values[-1]


class NumericalPatchEmbedding(nn.Module):
    """Create one BERT-dimensional numerical token per right-padded patch."""

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.hidden_size = int(hidden_size)
        self.num_patches = (self.seq_len + self.stride - self.patch_len) // self.stride + 1
        if self.num_patches <= 0:
            raise ValueError("The patch configuration produces no patches.")

        self.value_projection = nn.Linear(self.patch_len, self.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_patches, self.hidden_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.value_projection.bias)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, N, T], received {tuple(x.shape)}.")

        batch_size, n_vars, seq_len = x.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f"The model was initialized for seq_len={self.seq_len}, but received T={seq_len}."
            )

        x = F.pad(x, (0, self.stride), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.shape[2] != self.num_patches:
            raise RuntimeError(
                f"Expected {self.num_patches} patches, obtained {patches.shape[2]}."
            )

        patches = patches.contiguous().view(
            batch_size * n_vars, self.num_patches, self.patch_len
        )
        tokens = self.value_projection(patches)
        tokens = tokens + self.position_embedding.to(dtype=tokens.dtype)
        return self.dropout(tokens), n_vars


class LowRankVocabularyPrototypeMapper(nn.Module):
    """Construct a trainable prototype memory from the frozen BERT vocabulary."""

    def __init__(
        self,
        vocabulary_embeddings: torch.Tensor,
        num_prototypes: int = 1000,
        rank: int = 96,
    ) -> None:
        super().__init__()
        if vocabulary_embeddings.ndim != 2:
            raise ValueError("Vocabulary embeddings must have shape [V, D].")

        vocab_size, hidden_size = vocabulary_embeddings.shape
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_prototypes = int(num_prototypes)
        self.rank = int(rank)

        if self.num_prototypes <= 0 or self.rank <= 0:
            raise ValueError("num_prototypes and rank must be positive.")
        if self.num_prototypes > self.vocab_size:
            raise ValueError("num_prototypes cannot exceed the vocabulary size.")

        seed_token_ids = torch.linspace(
            0, self.vocab_size - 1, steps=self.num_prototypes
        ).round().long()
        base_prototypes = vocabulary_embeddings.detach()[seed_token_ids].clone()
        self.register_buffer("seed_token_ids", seed_token_ids, persistent=True)
        self.register_buffer("base_prototypes", base_prototypes, persistent=True)

        self.vocabulary_factor = nn.Parameter(torch.empty(self.vocab_size, self.rank))
        self.prototype_factor = nn.Parameter(
            torch.empty(self.num_prototypes, self.rank)
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.vocabulary_factor, mean=0.0, std=1.0 / math.sqrt(self.vocab_size)
        )
        nn.init.normal_(
            self.prototype_factor, mean=0.0, std=1.0 / math.sqrt(self.rank)
        )
        with torch.no_grad():
            self.residual_scale.fill_(0.1)

    def forward(self, vocabulary_embeddings: torch.Tensor) -> torch.Tensor:
        expected_shape = (self.vocab_size, self.hidden_size)
        if tuple(vocabulary_embeddings.shape) != expected_shape:
            raise ValueError(
                "The BERT vocabulary shape differs from the prototype-mapper configuration."
            )

        frozen_embeddings = vocabulary_embeddings.detach()
        vocabulary_basis = self.vocabulary_factor.transpose(0, 1) @ frozen_embeddings
        residual = self.prototype_factor @ vocabulary_basis
        return self.base_prototypes.to(dtype=residual.dtype) + self.residual_scale * residual


class Model(nn.Module):
    """ME-LLM model class for the repository forecasting framework."""

    def __init__(self, configs, patch_len: int = 16, stride: int = 8) -> None:
        super().__init__()

        self.task_name = str(_config_value(configs, "task_name", "long_term_forecast"))
        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise NotImplementedError(f"Unsupported task: {self.task_name}")

        self.pred_len = int(_config_value(configs, "pred_len"))
        self.seq_len = int(_config_value(configs, "seq_len"))
        self.patch_len = int(_config_value(configs, "patch_len", patch_len))
        self.stride = int(_config_value(configs, "stride", stride))
        self.top_k = int(_config_value(configs, "top_k", 5))
        self.num_prototypes = int(_config_value(configs, "num_prototypes", 1000))
        self.prototype_rank = int(_config_value(configs, "prototype_rank", 96))
        self.semantic_heads = int(
            _config_value(configs, "semantic_heads", _config_value(configs, "n_heads", 8))
        )
        self.max_prompt_tokens = int(_config_value(configs, "max_prompt_tokens", 176))
        self.max_prediction_length = int(
            _config_value(configs, "max_prediction_length", 720)
        )
        self.attention_chunk_size = int(
            _config_value(configs, "prototype_attention_chunk_size", 64)
        )
        dropout = float(_config_value(configs, "dropout", 0.1))

        if self.pred_len > self.max_prediction_length:
            raise ValueError(
                f"pred_len={self.pred_len} exceeds max_prediction_length={self.max_prediction_length}."
            )
        if self.attention_chunk_size <= 0:
            raise ValueError("prototype_attention_chunk_size must be positive.")

        requested_backbone = str(_config_value(configs, "llm_model", "BERT")).upper()
        if requested_backbone != "BERT":
            raise ValueError("ME-LLM uses a frozen BERT-base encoder; set llm_model='BERT'.")

        self.llm_model, self.tokenizer = self._load_bert(configs)
        self.d_llm = int(self.llm_model.config.hidden_size)
        self.max_position_embeddings = int(
            self.llm_model.config.max_position_embeddings
        )
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer must define a padding token.")

        for parameter in self.llm_model.parameters():
            parameter.requires_grad = False
        self.llm_model.eval()

        self.dataset_key, self.description = _resolve_dataset_description(configs)

        self.patch_embedding = NumericalPatchEmbedding(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            hidden_size=self.d_llm,
            dropout=dropout,
        )
        self.patch_nums = self.patch_embedding.num_patches

        vocabulary_embeddings = self.llm_model.get_input_embeddings().weight
        self.prototype_mapper = LowRankVocabularyPrototypeMapper(
            vocabulary_embeddings=vocabulary_embeddings,
            num_prototypes=self.num_prototypes,
            rank=self.prototype_rank,
        )
        self.prototype_attention = nn.MultiheadAttention(
            embed_dim=self.d_llm,
            num_heads=self.semantic_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.head_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(
            self.d_llm, self.max_prediction_length
        )
        self.normalize_layers = Normalize(
            int(_config_value(configs, "enc_in")), affine=False
        )
        self.last_runtime_info: Dict[str, float] = {}

        self._validate_configuration(configs)

    def _load_bert(self, configs) -> Tuple[BertModel, BertTokenizer]:
        source = str(_config_value(configs, "bert_path", "bert-base-uncased"))
        local_only = bool(_config_value(configs, "local_files_only", False))
        requested_layers = int(_config_value(configs, "llm_layers", 12))
        output_attentions = bool(_config_value(configs, "output_attention", False))

        bert_config = BertConfig.from_pretrained(
            source, local_files_only=local_only
        )
        if int(bert_config.num_hidden_layers) != requested_layers:
            raise ValueError(
                f"BERT at {source!r} has {bert_config.num_hidden_layers} layers; "
                f"the run configuration requests {requested_layers}."
            )
        bert_config.output_attentions = output_attentions
        bert_config.output_hidden_states = False
        bert_config.return_dict = True

        model = BertModel.from_pretrained(
            source,
            local_files_only=local_only,
            config=bert_config,
        )
        tokenizer = BertTokenizer.from_pretrained(
            source, local_files_only=local_only
        )
        return model, tokenizer

    def _validate_configuration(self, configs) -> None:
        expected_values = {
            "seq_len": (self.seq_len, 512),
            "patch_len": (self.patch_len, 16),
            "stride": (self.stride, 8),
            "patch_count": (self.patch_nums, 64),
            "prototype_count": (self.num_prototypes, 1000),
            "prototype_rank": (self.prototype_rank, 96),
            "semantic_heads": (self.semantic_heads, 8),
            "BERT_hidden_size": (self.d_llm, 768),
            "BERT_layers": (int(self.llm_model.config.num_hidden_layers), 12),
            "BERT_position_limit": (self.max_position_embeddings, 512),
            "maximum_prompt_tokens": (self.max_prompt_tokens, 176),
            "maximum_prediction_length": (self.max_prediction_length, 720),
        }
        mismatches = [
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in expected_values.items()
            if actual != expected
        ]
        if self.pred_len not in SUPPORTED_FORECAST_HORIZONS:
            mismatches.append(
                f"pred_len={self.pred_len} (expected one of {SUPPORTED_FORECAST_HORIZONS})"
            )
        if mismatches:
            raise ValueError("Configuration mismatch: " + "; ".join(mismatches))

        trainable = self.trainable_parameter_count()
        expected_trainable = 6_004_369
        tolerance = int(_config_value(configs, "parameter_count_tolerance", 50_000))
        if abs(trainable - expected_trainable) > tolerance:
            raise ValueError(
                f"Trainable parameter count is {trainable:,}; expected "
                f"{expected_trainable:,} ± {tolerance:,}."
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.llm_model.eval()
        return self

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def parameter_report(self) -> Dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": self.trainable_parameter_count(),
            "frozen": sum(
                parameter.numel()
                for parameter in self.parameters()
                if not parameter.requires_grad
            ),
        }

    def prototype_seed_tokens(self) -> List[str]:
        """Return the vocabulary strings used to initialize the prototype memory."""

        return self.tokenizer.convert_ids_to_tokens(
            self.prototype_mapper.seed_token_ids.detach().cpu().tolist()
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: Optional[torch.Tensor],
        x_mark_dec: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del mask
        output = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return output[:, -self.pred_len :, :]

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: Optional[torch.Tensor],
        x_mark_dec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec

        x_norm = self.normalize_layers(x_enc, "norm")
        batch_size, seq_len, n_vars = x_norm.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected input length {self.seq_len}, received {seq_len}.")

        per_variable = (
            x_norm.permute(0, 2, 1)
            .contiguous()
            .view(batch_size * n_vars, seq_len, 1)
        )
        prompts = self._build_prompts(per_variable)
        prompt_ids, prompt_mask, prompt_lengths = self._tokenize_prompts(
            prompts, x_norm.device
        )
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids)

        numerical_tokens, returned_n_vars = self.patch_embedding(
            x_norm.permute(0, 2, 1).contiguous()
        )
        if returned_n_vars != n_vars:
            raise RuntimeError("Patch embedding returned an inconsistent variable count.")

        prototypes = self.prototype_mapper(
            self.llm_model.get_input_embeddings().weight
        ).to(device=numerical_tokens.device, dtype=numerical_tokens.dtype)
        semantic_tokens = self._build_semantic_tokens(numerical_tokens, prototypes)
        paired_tokens = self.interleave_tokens(semantic_tokens, numerical_tokens)

        paired_tokens = paired_tokens.to(dtype=prompt_embeddings.dtype)
        encoder_embeddings = torch.cat((prompt_embeddings, paired_tokens), dim=1)
        paired_mask = torch.ones(
            paired_tokens.shape[:2],
            dtype=prompt_mask.dtype,
            device=prompt_mask.device,
        )
        encoder_attention_mask = torch.cat((prompt_mask, paired_mask), dim=1)

        encoder_width = int(encoder_embeddings.shape[1])
        if encoder_width > self.max_position_embeddings:
            raise RuntimeError(
                f"Encoder input has {encoder_width} tokens, exceeding the "
                f"{self.max_position_embeddings}-token positional limit."
            )

        contextualized = self.llm_model(
            inputs_embeds=encoder_embeddings,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        ).last_hidden_state

        prompt_width = int(prompt_embeddings.shape[1])
        patch_states = contextualized[
            :, prompt_width : prompt_width + 2 * self.patch_nums, :
        ]
        pair_states = patch_states.view(
            batch_size * n_vars, self.patch_nums, 2, self.d_llm
        )
        pooled_state = pair_states.mean(dim=2).mean(dim=1)
        all_horizons = self.output_projection(self.head_dropout(pooled_state))
        prediction = all_horizons[:, : self.pred_len]

        prediction = prediction.view(batch_size, n_vars, self.pred_len)
        prediction = prediction.permute(0, 2, 1).contiguous()
        prediction = self.normalize_layers(prediction, "denorm")

        with torch.no_grad():
            self.last_runtime_info = {
                "batch_size": float(batch_size),
                "variables": float(n_vars),
                "patches_per_variable": float(self.patch_nums),
                "numerical_tokens": float(self.patch_nums),
                "semantic_tokens": float(self.patch_nums),
                "prompt_tokens_mean": float(prompt_lengths.float().mean().item()),
                "prompt_tokens_max": float(prompt_lengths.max().item()),
                "encoder_tokens": float(encoder_width),
                "trainable_parameters": float(self.trainable_parameter_count()),
            }

        return prediction

    @staticmethod
    def interleave_tokens(
        semantic_tokens: torch.Tensor, numerical_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Return the local order [s1, p1, s2, p2, ...]."""

        if semantic_tokens.shape != numerical_tokens.shape:
            raise ValueError(
                "Semantic and numerical token tensors must have identical shapes."
            )
        if semantic_tokens.ndim != 3:
            raise ValueError(
                f"Expected [batch, patches, hidden], received {semantic_tokens.shape}."
            )
        batch, patches, hidden = semantic_tokens.shape
        return torch.stack(
            (semantic_tokens, numerical_tokens), dim=2
        ).reshape(batch, 2 * patches, hidden)

    def _build_semantic_tokens(
        self, numerical_tokens: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attend each numerical patch token to the shared prototype memory."""

        outputs: List[torch.Tensor] = []
        total = numerical_tokens.shape[0]
        for start in range(0, total, self.attention_chunk_size):
            stop = min(start + self.attention_chunk_size, total)
            query = numerical_tokens[start:stop]
            key_value = prototypes.unsqueeze(0).expand(stop - start, -1, -1)
            semantic, _ = self.prototype_attention(
                query=query,
                key=key_value,
                value=key_value,
                need_weights=False,
            )
            outputs.append(semantic)
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def _build_prompts(self, x_per_variable: torch.Tensor) -> List[str]:
        values = x_per_variable.squeeze(-1).float()
        minimum = values.amin(dim=1)
        maximum = values.amax(dim=1)
        median = values.median(dim=1).values
        slope = (values[:, -1] - values[:, 0]) / max(values.shape[1] - 1, 1)
        lags = self.calculate_lags(values)

        prompts: List[str] = []
        for min_value, max_value, median_value, slope_value, lag_values in zip(
            minimum.cpu().tolist(),
            maximum.cpu().tolist(),
            median.cpu().tolist(),
            slope.cpu().tolist(),
            lags.cpu().tolist(),
        ):
            if slope_value > 0:
                trend = "upward"
            elif slope_value < 0:
                trend = "downward"
            else:
                trend = "flat"

            prompt = (
                "<|start_prompt|>\n"
                f"Dataset: {self.description}\n"
                f"Forecast task: use the previous {self.seq_len} observations to "
                f"predict the next {self.pred_len} observations for the current variable.\n"
                "Observed-window statistics: "
                f"minimum {_format_number(float(min_value))}; "
                f"maximum {_format_number(float(max_value))}; "
                f"median {_format_number(float(median_value))}; "
                f"endpoint trend {trend}; strongest positive autocorrelation lags "
                f"{_format_lag_list(lag_values)}.\n"
                "Token layout: the following tokens are locally paired and ordered as "
                "semantic 1, patch 1, semantic 2, patch 2, ..., semantic n_p, patch n_p.\n"
                "<|end_prompt|>"
            )
            prompts.append(prompt)
        return prompts

    def _tokenize_prompts(
        self, prompts: Sequence[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        prompt_lengths = tokenized.attention_mask.sum(dim=1)
        longest_prompt = int(prompt_lengths.max().item())
        if longest_prompt > self.max_prompt_tokens:
            raise RuntimeError(
                f"The longest prompt has {longest_prompt} WordPiece tokens; "
                f"the configured limit is {self.max_prompt_tokens}."
            )

        padded_prompt_width = int(tokenized.input_ids.shape[1])
        total_width = padded_prompt_width + 2 * self.patch_nums
        if total_width > self.max_position_embeddings:
            raise RuntimeError(
                f"Prompt padding ({padded_prompt_width}) plus paired patch tokens "
                f"({2 * self.patch_nums}) gives {total_width}, exceeding the BERT "
                f"position limit of {self.max_position_embeddings}."
            )

        return (
            tokenized.input_ids.to(device),
            tokenized.attention_mask.to(device),
            prompt_lengths,
        )

    @torch.no_grad()
    def calculate_lags(self, x: torch.Tensor) -> torch.Tensor:
        """Return the top-k positive lag indices by centered autocorrelation."""

        if x.ndim == 3:
            if x.shape[-1] != 1:
                raise ValueError("A 3-D lag input must have final dimension 1.")
            x = x.squeeze(-1)
        if x.ndim != 2:
            raise ValueError(f"Expected [batch, time], received {tuple(x.shape)}.")

        values = x.float()
        time_steps = values.shape[1]
        if time_steps < 2:
            raise ValueError("At least two observations are required for lag analysis.")

        centered = values - values.mean(dim=1, keepdim=True)
        n_fft = 1 << (2 * time_steps - 1).bit_length()
        spectrum = torch.fft.rfft(centered, n=n_fft, dim=1)
        autocovariance = torch.fft.irfft(
            spectrum * torch.conj(spectrum), n=n_fft, dim=1
        )[:, :time_steps]
        denominator = centered.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
        autocorrelation = autocovariance[:, 1:time_steps] / denominator

        k = min(self.top_k, time_steps - 1)
        strongest = torch.topk(
            autocorrelation, k=k, dim=1, largest=True
        ).indices + 1
        return strongest.sort(dim=1).values
