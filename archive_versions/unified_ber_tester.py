#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import archive_versions.experiment1 as exp


CONSTELLATION = torch.tensor(
    [
        [-0.948683, -0.948683], [-0.948683, -0.316228], [-0.948683, 0.316228], [-0.948683, 0.948683],
        [-0.316228, -0.948683], [-0.316228, -0.316228], [-0.316228, 0.316228], [-0.316228, 0.948683],
        [0.316228, -0.948683], [0.316228, -0.316228], [0.316228, 0.316228], [0.316228, 0.948683],
        [0.948683, -0.948683], [0.948683, -0.316228], [0.948683, 0.316228], [0.948683, 0.948683],
    ],
    dtype=torch.float32,
)

BIT_LABELS = torch.tensor(
    [
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0],
        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 0],
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0],
        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 1, 0],
    ],
    dtype=torch.uint8,
)


@dataclass
class ReferenceConfig:
    source: str
    data_size: Optional[int]
    data_files: Optional[int]
    train_portion: Optional[float]
    train_files: Optional[int]
    test_files: Optional[int]
    radius: Optional[int]
    batch_size: Optional[int]
    lrate: Optional[float]
    seed_mode: str
    precision: str
    gpu_ctx: str


def _extract_assignment(text: str, key: str, cast):
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\n#]+)", text, flags=re.MULTILINE)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        return cast(raw)
    except Exception:
        return None


def parse_cnn_complex(path: Path) -> ReferenceConfig:
    text = path.read_text(encoding="utf-8")
    radius = _extract_assignment(text, "radius", int)
    if radius is None:
        arr = re.search(r"radius_array\s*=\s*\[([^\]]+)\]", text)
        if arr:
            vals = [v.strip() for v in arr.group(1).split(",") if v.strip()]
            if vals:
                try:
                    radius = int(vals[0])
                except Exception:
                    radius = None
    data_size = _extract_assignment(text, "data_size", int)
    data_files = _extract_assignment(text, "data_files", int)
    train_portion = _extract_assignment(text, "train_portion", float)
    train_files = _extract_assignment(text, "train_files", int)
    test_files = _extract_assignment(text, "test_files", int)
    minibatch_multiplier = _extract_assignment(text, "minibatch_multiplier", int)
    batch_size = _extract_assignment(text, "batch_size", int)
    if train_files is None and data_files is not None and train_portion is not None:
        train_files = int(data_files * train_portion)
    if test_files is None and data_files is not None and train_files is not None:
        test_files = data_files - train_files
    if batch_size is None and data_size is not None and minibatch_multiplier is not None:
        batch_size = data_size * minibatch_multiplier

    return ReferenceConfig(
        source=path.name,
        data_size=data_size,
        data_files=data_files,
        train_portion=train_portion,
        train_files=train_files,
        test_files=test_files,
        radius=radius,
        batch_size=batch_size,
        lrate=_extract_assignment(text, "lrate", float),
        seed_mode="random (randint + mx.random.seed)",
        precision="float32 by default (float64 optional via swap_64_to_128_complex)",
        gpu_ctx="mxnet gpu(0)",
    )


def parse_mxnet_notebook(path: Path) -> ReferenceConfig:
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []) if cell.get("cell_type") == "code")
    data_size = _extract_assignment(code, "data_size", int)
    data_files = _extract_assignment(code, "data_files", int)
    train_portion = _extract_assignment(code, "train_portion", float)
    train_files = _extract_assignment(code, "train_files", int)
    test_files = _extract_assignment(code, "test_files", int)
    minibatch_multiplier = _extract_assignment(code, "minibatch_multiplier", int)
    batch_size = _extract_assignment(code, "batch_size", int)
    if train_files is None and data_files is not None and train_portion is not None:
        train_files = int(data_files * train_portion)
    if test_files is None and data_files is not None and train_files is not None:
        test_files = data_files - train_files
    if batch_size is None and data_size is not None and minibatch_multiplier is not None:
        batch_size = data_size * minibatch_multiplier

    return ReferenceConfig(
        source=path.name,
        data_size=data_size,
        data_files=data_files,
        train_portion=train_portion,
        train_files=train_files,
        test_files=test_files,
        radius=_extract_assignment(code, "radius", int),
        batch_size=batch_size,
        lrate=_extract_assignment(code, "lrate", float),
        seed_mode="random (randint + mx.random.seed)",
        precision="float32 by default (float64 optional via swap_64_to_128_complex)",
        gpu_ctx="mxnet gpu(0)",
    )


def current_experiment_config() -> Dict[str, object]:
    return {
        "source": "experiment1.py",
        "data_size": None,
        "data_files": exp.Config.MAX_FILES,
        "train_portion": None,
        "train_files": None,
        "test_files": None,
        "radius": exp.Config.CONTEXT_K,
        "batch_size": exp.Config.BATCH_SIZE,
        "lrate": exp.Config.LEARNING_RATE,
        "seed_mode": "not fixed in file (gap)",
        "precision": "float32 + cuda autocast fp16",
        "gpu_ctx": str(exp.Config.DEVICE),
        "split": {
            "val_split_ratio": exp.Config.VAL_SPLIT_RATIO,
            "test_split_ratio": exp.Config.TEST_SPLIT_RATIO,
            "train_file_indices": exp.Config.TRAIN_FILE_INDICES,
            "val_file_indices": exp.Config.VAL_FILE_INDICES,
            "test_file_indices": exp.Config.TEST_FILE_INDICES,
        },
    }


def discover_local_indices() -> Tuple[Path, List[int]]:
    pattern = re.compile(r"^(?:S_)?Symbols_1m_1ch_PR_(\d+)\.csv$", re.IGNORECASE)
    for candidate_dir in exp.Config.DATA_DIR_CANDIDATES:
        if not candidate_dir.exists() or not candidate_dir.is_dir():
            continue
        found = []
        for p in candidate_dir.iterdir():
            m = pattern.match(p.name)
            if p.is_file() and m:
                found.append(int(m.group(1)))
        if found:
            return candidate_dir, sorted(set(found))
    raise FileNotFoundError("No symbol files found in DATA_DIR_CANDIDATES")


def build_reference_split(available_indices: List[int], data_files: int, train_portion: float) -> Tuple[List[int], List[int]]:
    use = available_indices[:data_files]
    train_files = int(data_files * train_portion)
    return use[:train_files], use[train_files:]


def load_files(base_dir: Path, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    tx_chunks: List[np.ndarray] = []
    rx_chunks: List[np.ndarray] = []
    for idx in indices:
        p = base_dir / f"Symbols_1m_1ch_PR_{idx}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        arr = pd.read_csv(p, header=None, dtype=np.float32).to_numpy(copy=False)
        tx_chunks.append(arr[:, 0:2].copy())
        rx_chunks.append(arr[:, 2:4].copy())
    tx = np.concatenate(tx_chunks, axis=0)
    rx = np.concatenate(rx_chunks, axis=0)
    return tx, rx


def symbols_to_classes_np(symbols: np.ndarray) -> np.ndarray:
    s = torch.from_numpy(symbols.astype(np.float32, copy=False))
    diff = s.unsqueeze(1) - CONSTELLATION.unsqueeze(0)
    dist = torch.sum(diff * diff, dim=2)
    return torch.argmin(dist, dim=1).cpu().numpy().astype(np.int64, copy=False)


def ber_from_classes_np(tx_classes: np.ndarray, rx_classes: np.ndarray) -> float:
    tx_bits = BIT_LABELS[torch.from_numpy(tx_classes)]
    rx_bits = BIT_LABELS[torch.from_numpy(rx_classes)]
    return (tx_bits != rx_bits).float().mean().item()


def instantiate_model(model_type: str) -> torch.nn.Module:
    if model_type == "lstm":
        return exp.LSTMRxEqualizer(input_dim=2)
    if model_type in {"hybrid", "cnn_lstm"}:
        return exp.HybridCNN_LSTM_Equalizer(input_dim=2)
    if model_type == "mlp":
        return exp.MLPRxEqualizer(input_dim=2)
    raise ValueError(f"Unsupported model type: {model_type}")


def infer_model_type(checkpoint: Path, payload: dict) -> str:
    lower = checkpoint.name.lower()
    if "lstm" in lower and "hybrid" not in lower:
        return "lstm"
    if "hybrid" in lower or "cnn_lstm" in lower:
        return "hybrid"
    if "mlp" in lower:
        return "mlp"
    arch = str(payload.get("model_architecture", "")).lower()
    if "hybrid" in arch:
        return "hybrid"
    if "mlp" in arch:
        return "mlp"
    return "lstm"


def evaluate_checkpoint(
    checkpoint_path: Path,
    rx_test: np.ndarray,
    tx_test: np.ndarray,
    eval_k: int,
    batch_size: int,
    device: torch.device,
    shared_mean: torch.Tensor,
    shared_std: torch.Tensor,
) -> Dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_type = infer_model_type(checkpoint_path, payload)
    cfg = payload.get("config", {})
    ckpt_k = int(cfg.get("CONTEXT_K", payload.get("context_k", eval_k)))
    if ckpt_k != eval_k:
        raise ValueError(f"{checkpoint_path.name}: CONTEXT_K={ckpt_k}, expected {eval_k}")

    model = instantiate_model(model_type).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    mean = payload.get("normalization_mean")
    std = payload.get("normalization_std")
    if mean is None or std is None:
        mean = shared_mean
        std = shared_std
    else:
        mean = mean.float().cpu()
        std = std.float().cpu()
    std = std.clone()
    std[std == 0] = 1.0

    rx_t = torch.from_numpy(rx_test).float()
    windows = rx_t.unfold(0, 2 * eval_k + 1, 1).permute(0, 2, 1).contiguous()
    windows = (windows - mean) / std
    x = windows.reshape(windows.shape[0], -1)

    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size].to(device, non_blocking=True)
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu())
    pred_classes = torch.cat(preds, dim=0).numpy()

    tx_classes = symbols_to_classes_np(tx_test[eval_k: len(tx_test) - eval_k])
    ber = ber_from_classes_np(tx_classes, pred_classes)
    acc = float((tx_classes == pred_classes).mean())

    return {
        "model": checkpoint_path.name,
        "model_type": model_type,
        "context_k": eval_k,
        "windows": int(len(pred_classes)),
        "equalized_ber": ber,
        "symbol_accuracy": acc,
    }


def write_audit_markdown(
    out_path: Path,
    cnn_cfg: ReferenceConfig,
    nb_cfg: ReferenceConfig,
    current_cfg: Dict[str, object],
    split_train: List[int],
    split_test: List[int],
    distribution: Dict[str, object],
):
    lines = []
    lines.append("# BER Protocol Audit")
    lines.append("")
    lines.append("| Parameter | CNN_complex_v1.py | MXNet_Complex_FCNN_Conv_GPU_v1.ipynb | experiment1.py |")
    lines.append("|---|---:|---:|---:|")
    for key in ["data_files", "train_portion", "train_files", "test_files", "data_size", "radius", "batch_size", "lrate"]:
        lines.append(
            f"| {key} | {getattr(cnn_cfg, key)} | {getattr(nb_cfg, key)} | {current_cfg.get(key)} |"
        )
    lines.append(f"| seed_mode | {cnn_cfg.seed_mode} | {nb_cfg.seed_mode} | {current_cfg.get('seed_mode')} |")
    lines.append(f"| precision | {cnn_cfg.precision} | {nb_cfg.precision} | {current_cfg.get('precision')} |")
    lines.append(f"| gpu_ctx | {cnn_cfg.gpu_ctx} | {nb_cfg.gpu_ctx} | {current_cfg.get('gpu_ctx')} |")
    lines.append("")
    lines.append("## Unified Reference Split Used")
    lines.append(f"- train files: {split_train}")
    lines.append(f"- test files: {split_test}")
    lines.append("")
    lines.append("## Test Distribution Validation")
    for k, v in distribution.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Explicit Gaps")
    lines.append("- seed is random in both external files (`randint(...)`), so runs are not inherently deterministic.")
    lines.append("- SNR/channel are encoded only via source file paths, not explicit runtime parameters in code.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit and run unified BER testing protocol.")
    parser.add_argument("--reference", choices=["cnn", "notebook"], default="cnn")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--output-csv", default="unified_ber_results.csv")
    parser.add_argument("--audit-md", default="ber_protocol_audit.md")
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    args = parser.parse_args()

    root = Path(".")
    cnn_cfg = parse_cnn_complex(root / "CNN_complex_v1.py")
    nb_cfg = parse_mxnet_notebook(root / "MXNet_Complex_FCNN_Conv_GPU_v1.ipynb")
    current_cfg = current_experiment_config()
    ref_cfg = cnn_cfg if args.reference == "cnn" else nb_cfg

    if ref_cfg.data_files is None or ref_cfg.train_portion is None or ref_cfg.radius is None:
        raise ValueError(f"Missing required protocol params in {ref_cfg.source}")

    base_dir, available = discover_local_indices()
    if len(available) < ref_cfg.data_files:
        raise ValueError(f"Not enough local files ({len(available)}) for reference data_files={ref_cfg.data_files}")

    split_train, split_test = build_reference_split(available, ref_cfg.data_files, ref_cfg.train_portion)
    tx_train, rx_train = load_files(base_dir, split_train)
    tx_test, rx_test = load_files(base_dir, split_test)

    sample_size = int(ref_cfg.data_size - 2 * ref_cfg.radius) if ref_cfg.data_size else (len(tx_test) // len(split_test) - 2 * ref_cfg.radius)
    observed_per_file = int(len(tx_test) // max(len(split_test), 1))
    observed_cropped_total = len(split_test) * (observed_per_file - 2 * ref_cfg.radius)
    distribution = {
        "base_dir": str(base_dir),
        "reference_source": ref_cfg.source,
        "test_files_count": len(split_test),
        "test_samples_total": int(len(tx_test)),
        "test_samples_per_file_expected": sample_size,
        "test_samples_per_file_observed": observed_per_file,
        "center_cropped_symbols_expected_total": int(len(split_test) * sample_size),
        "center_cropped_symbols_observed_total": int(observed_cropped_total),
    }

    write_audit_markdown(Path(args.audit_md), cnn_cfg, nb_cfg, current_cfg, split_train, split_test, distribution)

    baseline_before = ber_from_classes_np(symbols_to_classes_np(tx_test), symbols_to_classes_np(rx_test))
    print(f"[protocol] reference={ref_cfg.source} | files={ref_cfg.data_files} | train={len(split_train)} | test={len(split_test)}")
    print(f"[validation] test distribution: {distribution}")
    print(f"[baseline] BER before equalizer (reference rule): {baseline_before:.6e}")

    if not args.checkpoints:
        print("[info] no checkpoints provided; audit markdown generated.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_mean = torch.from_numpy(rx_train).float().mean(dim=0, keepdim=True)
    shared_std = torch.from_numpy(rx_train).float().std(dim=0, keepdim=True)
    shared_std[shared_std == 0] = 1.0

    rows = []
    for ckpt in [Path(p) for p in args.checkpoints]:
        result = evaluate_checkpoint(
            checkpoint_path=ckpt,
            rx_test=rx_test,
            tx_test=tx_test,
            eval_k=int(ref_cfg.radius),
            batch_size=args.eval_batch_size,
            device=device,
            shared_mean=shared_mean,
            shared_std=shared_std,
        )
        result["baseline_ber"] = baseline_before
        result["improvement_rel_pct"] = (1.0 - result["equalized_ber"] / baseline_before) * 100.0 if baseline_before > 0 else np.nan
        rows.append(result)
        print(
            f"[result] {result['model']}: BER={result['equalized_ber']:.6e} "
            f"Acc={result['symbol_accuracy']:.4%} RelImp={result['improvement_rel_pct']:.2f}%"
        )

    df = pd.DataFrame(rows).sort_values("equalized_ber", ascending=True).reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[saved] {args.output_csv}")


if __name__ == "__main__":
    main()
