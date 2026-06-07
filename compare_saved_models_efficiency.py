import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

import ber_equalization as be


MODEL_NAMES = [
    "mlp",
    "cnn",
    "lstm",
    "hybrid",
    "transformer",
    "tcn",
    "mamba",
    "efficient_kan_baseline",
    "efficient_kan_residual",
    "efficient_kan_features",
    "cnn_kan",
    "kan_classifier",
    "complex_fastkan",
    "fastkan_classifier",
    "complex_fastkan_classifier",
    "complex_cnn",
    "complex_lstm",
    "complex_cnn_lstm",
    "complex_dbp_seqstat",
]

CONFIG_SNAPSHOT_KEYS = [
    "CONTEXT_K",
    "SEQ_LEN",
    "HIDDEN_DIM",
    "LSTM_HIDDEN",
    "LSTM_LAYERS",
    "BIDIRECTIONAL",
    "TRANSFORMER_DIM",
    "TRANSFORMER_LAYERS",
    "TRANSFORMER_HEADS",
    "TRANSFORMER_FF_DIM",
    "TCN_HIDDEN_DIM",
    "TCN_LAYERS",
    "TCN_KERNEL_SIZE",
    "TCN_DILATIONS",
    "MAMBA_DIM",
    "MAMBA_LAYERS",
    "MAMBA_D_STATE",
    "MAMBA_D_CONV",
    "MAMBA_EXPAND",
    "COMPLEX_BLOCK_CHANNELS",
    "COMPLEX_KERNEL_SIZES",
    "COMPLEX_HEAD_DIM",
    "COMPLEX_TEMPORAL_DIM",
    "COMPLEX_TEMPORAL_DILATIONS",
    "COMPLEX_LIGHT_CHANNELS",
    "COMPLEX_LIGHT_DILATIONS",
    "COMPLEX_LIGHT_KERNEL_SIZE",
    "COMPLEX_SEQ_DIM",
    "COMPLEX_LSTM_HIDDEN",
    "COMPLEX_LSTM_LAYERS",
    "DBP_NUM_STEPS",
    "DBP_KERNEL_SIZE",
    "DBP_FINAL_KERNEL_SIZE",
    "DBP_USE_FINAL_FILTER",
    "DBP_USE_SYMMETRIC_FILTER",
    "DBP_USE_SYMMETRIC_NONLINEAR",
    "DBP_NL_MEMORY",
    "DBP_SEQSTAT_DIM",
    "EFFICIENT_KAN_HIDDEN_DIM",
    "EFFICIENT_KAN_LAYERS",
    "EFFICIENT_KAN_GRID_SIZE",
    "EFFICIENT_KAN_SPLINE_ORDER",
    "EFFICIENT_KAN_GRID_EPS",
    "EFFICIENT_KAN_GRID_RANGE",
    "FASTKAN_HIDDEN_DIM",
    "FASTKAN_LAYERS",
    "FASTKAN_NUM_GRIDS",
    "FASTKAN_GRID_MIN",
    "FASTKAN_GRID_MAX",
    "MLP_LAYERS",
]

DEFAULT_CONFIG = {key: copy_value for key, copy_value in ((key, getattr(be.Config, key)) for key in CONFIG_SNAPSHOT_KEYS)}


def reset_architecture_config() -> None:
    for key, value in DEFAULT_CONFIG.items():
        setattr(be.Config, key, value.copy() if isinstance(value, list) else value)


def log(message: str) -> None:
    print(message, flush=True)


def strip_compile_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return state


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError("checkpoint is not a state_dict")
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return strip_compile_prefix(state)


def count_indexed_layers(state: Dict[str, torch.Tensor], prefix: str) -> int:
    indices = set()
    for key in state:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        part = suffix.split(".", 1)[0]
        if part.isdigit():
            indices.add(int(part))
    return max(indices) + 1 if indices else 0


class LegacyMLPEqualizer(nn.Module):
    def __init__(self, state: Dict[str, torch.Tensor]):
        super().__init__()
        if "skip.weight" not in state:
            raise ValueError("legacy MLP checkpoint has no skip layer")
        input_dim = int(state["skip.weight"].shape[1])
        self.skip = nn.Linear(input_dim, 2)
        layer_indices = sorted(
            {
                int(key.split(".")[1])
                for key, value in state.items()
                if key.startswith("net.") and key.endswith(".weight") and value.dim() == 2
            }
        )
        modules: List[nn.Module] = []
        for idx in layer_indices:
            weight = state[f"net.{idx}.weight"]
            in_dim = int(weight.shape[1])
            out_dim = int(weight.shape[0])
            modules.append(nn.Linear(in_dim, out_dim))
            if f"net.{idx + 1}.weight" in state and state[f"net.{idx + 1}.weight"].dim() == 1:
                modules.append(nn.LayerNorm(out_dim))
                modules.append(nn.GELU())
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.net(x)


def has_encoder_only_complex_state(model_name: str, state: Dict[str, torch.Tensor]) -> bool:
    return (
        model_name == "complex_cnn"
        and any(key.startswith("stem.") for key in state)
        and not any(key.startswith("head.") for key in state)
        and not any(key.startswith("encoder.") for key in state)
    )


def should_skip_before_inference(model_name: str, state: Dict[str, torch.Tensor]) -> Optional[str]:
    if has_encoder_only_complex_state(model_name, state):
        return "checkpoint contains ComplexFeatureEncoder weights only, no equalizer head for BER inference"
    if model_name == "complex_dbp_seqstat":
        return (
            "automatic comparison skips complex_dbp_seqstat: DBP frontend requires a large context "
            "and checkpoint state_dict does not store the original CONTEXT_K"
        )
    return None


def infer_context_from_flat_input(input_dim: int) -> None:
    if input_dim % be.Config.INPUT_DIM != 0:
        return
    seq_len = input_dim // be.Config.INPUT_DIM
    if seq_len > 0 and seq_len % 2 == 1:
        be.Config.SEQ_LEN = seq_len
        be.Config.CONTEXT_K = seq_len // 2


def infer_config(model_name: str, state: Dict[str, torch.Tensor]) -> None:
    """Recover the main architecture settings encoded in checkpoint tensor shapes."""
    if "skip.weight" in state:
        infer_context_from_flat_input(int(state["skip.weight"].shape[1]))

    if "pos_embedding" in state and state["pos_embedding"].dim() == 3:
        seq_len = int(state["pos_embedding"].shape[1])
        if seq_len % 2 == 1:
            be.Config.SEQ_LEN = seq_len
            be.Config.CONTEXT_K = seq_len // 2

    kan_prefix = "kan.layers."
    kan_layer_count = count_indexed_layers(state, kan_prefix)
    if kan_layer_count:
        first_base = state.get("kan.layers.0.base_weight")
        first_spline = state.get("kan.layers.0.spline_weight")
        first_grid = state.get("kan.layers.0.grid")
        if first_base is not None:
            infer_context_from_flat_input(int(first_base.shape[1]))
            if kan_layer_count > 1:
                be.Config.EFFICIENT_KAN_HIDDEN_DIM = int(first_base.shape[0])
                be.Config.EFFICIENT_KAN_LAYERS = kan_layer_count - 1
        if first_spline is not None and first_grid is not None:
            basis_count = int(first_spline.shape[2])
            grid_point_count = int(first_grid.shape[1])
            order = grid_point_count - basis_count - 1
            if order < 1:
                raise ValueError("cannot infer EfficientKAN spline order from checkpoint shapes")
            be.Config.EFFICIENT_KAN_SPLINE_ORDER = order
            be.Config.EFFICIENT_KAN_GRID_SIZE = basis_count - order

    head_norm = state.get("head.input_norm.weight")
    if model_name == "fastkan_classifier" and head_norm is not None:
        infer_context_from_flat_input(int(head_norm.numel()))

    fastkan_prefix = "head.layers."
    fastkan_layers = count_indexed_layers(state, fastkan_prefix)
    if fastkan_layers:
        base_weight = state.get("head.layers.0.base_linear.weight")
        spline_weight = state.get("head.layers.0.spline_linear.weight")
        if base_weight is not None:
            be.Config.FASTKAN_HIDDEN_DIM = int(base_weight.shape[0])
            be.Config.FASTKAN_LAYERS = fastkan_layers
        if base_weight is not None and spline_weight is not None:
            in_features = int(base_weight.shape[1])
            expanded_features = int(spline_weight.shape[1])
            if in_features > 0 and expanded_features % in_features == 0:
                be.Config.FASTKAN_NUM_GRIDS = expanded_features // in_features

    if model_name == "tcn" and "stem.0.weight" in state:
        be.Config.TCN_HIDDEN_DIM = int(state["stem.0.weight"].shape[0])
        be.Config.TCN_LAYERS = count_indexed_layers(state, "blocks.")
        if "head.0.weight" in state:
            be.Config.HIDDEN_DIM = int(state["head.0.weight"].shape[0])

    if model_name == "mamba" and "input_proj.0.weight" in state:
        be.Config.MAMBA_DIM = int(state["input_proj.0.weight"].shape[0])
        be.Config.MAMBA_LAYERS = count_indexed_layers(state, "blocks.")

    if model_name == "lstm" and "lstm.weight_hh_l0" in state:
        be.Config.LSTM_HIDDEN = int(state["lstm.weight_hh_l0"].shape[1])
        layer_indices = {
            int(key.split("lstm.weight_ih_l", 1)[1].split("_", 1)[0])
            for key in state
            if key.startswith("lstm.weight_ih_l") and "reverse" not in key
        }
        if layer_indices:
            be.Config.LSTM_LAYERS = max(layer_indices) + 1
        if "classifier.0.weight" in state:
            be.Config.HIDDEN_DIM = int(state["classifier.0.weight"].shape[0])

    if model_name == "mlp" and "net.0.weight" in state:
        be.Config.HIDDEN_DIM = int(state["net.0.weight"].shape[0])
        linear_keys = [
            key
            for key, value in state.items()
            if key.startswith("net.") and key.endswith(".weight") and value.dim() == 2
        ]
        be.Config.MLP_LAYERS = max(len(linear_keys) - 1, 1)

    if model_name == "cnn" and "cnn.0.weight" in state:
        be.Config.HIDDEN_DIM = int(state["cnn.0.weight"].shape[0])

    if model_name in {"hybrid", "cnn_lstm"} and "lstm.weight_hh_l0" in state:
        be.Config.LSTM_HIDDEN = int(state["lstm.weight_hh_l0"].shape[1])
        layer_indices = {
            int(key.split("lstm.weight_ih_l", 1)[1].split("_", 1)[0])
            for key in state
            if key.startswith("lstm.weight_ih_l") and "reverse" not in key
        }
        if layer_indices:
            be.Config.LSTM_LAYERS = max(layer_indices) + 1
        be.Config.BIDIRECTIONAL = any("reverse" in key for key in state if key.startswith("lstm."))
    if model_name in {"hybrid", "cnn_lstm"} and "classifier.0.weight" in state:
        be.Config.HIDDEN_DIM = int(state["classifier.0.weight"].shape[0])

    if model_name == "transformer" and "input_proj.weight" in state:
        be.Config.TRANSFORMER_DIM = int(state["input_proj.weight"].shape[0])
        be.Config.TRANSFORMER_LAYERS = count_indexed_layers(state, "blocks.")
        if "blocks.0.ff_out.weight" in state:
            be.Config.TRANSFORMER_FF_DIM = int(state["blocks.0.ff_out.weight"].shape[1])
        if "regressor.0.weight" in state:
            be.Config.HIDDEN_DIM = int(state["regressor.0.weight"].shape[0])



def checkpoint_name(path: Path) -> str:
    suffix = "_final.pth"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return path.name[: -len(suffix)]


def discover_checkpoints(weights_dir: Path) -> List[Tuple[str, Path]]:
    found = {checkpoint_name(path): path for path in weights_dir.glob("*_final.pth")}
    ordered: List[Tuple[str, Path]] = []
    for name in MODEL_NAMES:
        if name in found:
            ordered.append((name, found.pop(name)))
    ordered.extend(sorted(found.items()))
    return ordered


def save_plots(results: pd.DataFrame, output_dir: Path) -> None:
    if results.empty:
        return
    ranked = results.sort_values("efficiency_score", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(ranked["model_type"], ranked["efficiency_score"], color="#0f9d87")
    ax.set_title("BER-Speed Efficiency Score")
    ax.set_ylabel("Higher is better")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "saved_models_efficiency_score.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in results.iterrows():
        ax.scatter(row["efficiency_batch_time_sec"], row["equalized_ber"], s=75)
        ax.annotate(row["model_type"], (row["efficiency_batch_time_sec"], row["equalized_ber"]), fontsize=8)
    ax.set_title("BER vs inference time for 16000 symbols")
    ax.set_xlabel("Batch inference time, seconds")
    ax.set_ylabel("Test BER")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "saved_models_ber_vs_time.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved equalizer checkpoints by BER-speed efficiency.")
    parser.add_argument("--weights-dir", type=Path, default=be.Config.OUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16000)
    parser.add_argument("--power", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    weights_dir = args.weights_dir.resolve()
    output_dir = (args.output_dir or weights_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    be.Config.EFFICIENCY_BATCH_SIZE = args.batch_size
    be.Config.EFFICIENCY_SCORE_POWER = args.power
    be.Config.EFFICIENCY_TIMING_WARMUP = args.warmup
    be.Config.EFFICIENCY_TIMING_REPEATS = args.repeats
    be.Config.COMPUTE_PER_FILE_METRICS = False

    checkpoints = discover_checkpoints(weights_dir)
    if not checkpoints:
        log(f"No *_final.pth checkpoints found in {weights_dir}")
        return

    log(f"Device: {be.Config.DEVICE}")
    log(f"Found {len(checkpoints)} checkpoints in {weights_dir}")
    data_cache: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for model_name, checkpoint_path in checkpoints:
        log(f"\n[{model_name}] loading {checkpoint_path.name}")
        try:
            reset_architecture_config()
            state = load_state(checkpoint_path)
            skip_reason = should_skip_before_inference(model_name, state)
            if skip_reason is not None:
                raise ValueError(skip_reason)
            infer_config(model_name, state)
            context_k = int(be.Config.CONTEXT_K)
            if context_k not in data_cache:
                log(f"Preparing test data for CONTEXT_K={context_k}")
                data_cache[context_k] = be.prepare_data()
            data = data_cache[context_k]
            if model_name == "mlp":
                model = LegacyMLPEqualizer(state).to(be.Config.DEVICE)
            else:
                model = be.make_model(model_name)
            model.load_state_dict(state, strict=True)
            metrics = be.compute_test_metrics(model, data)
            metrics = be.add_efficiency_metrics(model, data, metrics)
            rows.append(
                {
                    "rank": 0,
                    "model_type": model_name,
                    "checkpoint": str(checkpoint_path),
                    "context_k": context_k,
                    "seq_len": int(be.Config.SEQ_LEN),
                    "trainable_params": be.count_trainable_parameters(model),
                    **metrics,
                }
            )
            log(
                f"[{model_name}] BER={metrics['equalized_ber']:.6e} | "
                f"t{metrics['efficiency_batch_size']}={metrics['efficiency_batch_time_sec']:.6f}s | "
                f"score={metrics['efficiency_score']:.3f}"
            )
            model.to("cpu")
            del model
            if be.Config.DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            skipped.append({"model_type": model_name, "checkpoint": str(checkpoint_path), "reason": reason})
            log(f"[{model_name}] skipped: {reason}")

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values("efficiency_score", ascending=False).reset_index(drop=True)
        results["rank"] = results.index + 1
        results.to_csv(output_dir / "saved_models_efficiency_comparison.csv", index=False)
        save_plots(results, output_dir)
        log("\nRanking:")
        log(
            results[
                [
                    "rank",
                    "model_type",
                    "equalized_ber",
                    "efficiency_batch_time_sec",
                    "efficiency_score",
                    "trainable_params",
                ]
            ].to_string(index=False)
        )
    pd.DataFrame(skipped, columns=["model_type", "checkpoint", "reason"]).to_csv(
        output_dir / "saved_models_efficiency_skipped.csv", index=False
    )
    log(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
