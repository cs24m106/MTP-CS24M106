#!/usr/bin/env python3
"""
eval_comparator.py — Compare multiple ICCGAN training runs side-by-side.
Usage:
    python eval_comparator.py run_dir1 run_dir2 [run_dir3 ...]
    python eval_comparator.py run_dir1 run_dir2 --output comparison.png
    python eval_comparator.py run_dir1 run_dir2 --no-plots
"""
import argparse
import ast
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TextIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CSV_NAME = "training_metrics.csv"
CMDS_NAME = "cmds.txt"
COLORS = [
    # Professional colorblind-friendly palette (Okabe-Ito inspired)
    "#0072B2",  # Blue - primary
    "#D55E00",  # Vermilion - secondary  
    "#009E73",  # Bluish Green - tertiary
    "#CC79A7",  # Reddish Purple - quaternary
    "#56B4E9",  # Sky Blue - light accent
    "#F0E442",  # Yellow - highlight (use with dark background)
    "#332288",  # Indigo - dark but not black
    "#E69F00",  # Orange - warm accent
    "#999999",  # Gray - neutral
    "#88CCEE",  # Light Blue - soft accent
]

# Line style configurations for better visual hierarchy
LINE_CONFIG = {
    "raw_alpha": 0.06,      # Very transparent raw data (background texture)
    "raw_lw": 0.28,         # Thinner raw lines
    "smooth_alpha": 0.88,   # High visibility smoothed lines
    "smooth_lw": 1.05,      # Slightly thinner smoothed lines
    "marker_size": 6,       # Smaller min/max markers
    "marker_alpha": 0.62,   # Semi-transparent markers
    "connect_alpha": 0.12,  # Very subtle min/max connection lines
    "connect_lw": 0.5,      # Thinner connection lines
}

# Grid and axis styling
GRID_CONFIG = {
    "alpha": 0.20,          # Subtle grid
    "linewidth": 0.4,       # Fine grid lines
    "save_interval_alpha": 0.35,  # More visible period markers
    "save_interval_lw": 0.7,
}

# Required parameters that MUST be found in cmds.txt
REQUIRED_TRAIN_PARAMS = ['horizon', 'num_envs', 'save_interval', 'log_interval', 'sym_loss_coeff']
REQUIRED_ENV_PARAMS = ['max_cycles', 'steps_per_cycle', 'loop_phase_obs', 'fps']

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def _is_primitive(value: Any) -> bool:
    """Check if value is a primitive type (int, float, str, bool, None)"""
    return isinstance(value, (int, float, str, bool)) or value is None

def _extract_primitives_from_dict(d: dict) -> dict:
    """Extract only primitive values from dict, ignoring nested structures"""
    result = {}
    for k, v in d.items():
        if _is_primitive(v):
            result[k] = v
        # Skip lists, dicts, numpy arrays, tensors, etc.
    return result

def _safe_eval_dict(text: str) -> dict:
    """Safely evaluate a dict string, handling numpy type representations"""
    # Replace numpy type representations with plain values
    text = re.sub(r'np\.float64\(([\d.\-]+)\)', r'\1', text)
    text = re.sub(r'np\.int64\(([\d\-]+)\)', r'\1', text)
    text = re.sub(r'np\.bool_\((True|False)\)', r'\1', text)
    text = re.sub(r'nan', 'None', text)
    text = re.sub(r'inf', '1e308', text)
    
    try:
        result = ast.literal_eval(text)
        return _extract_primitives_from_dict(result)
    except Exception as e:
        print(f"  [!] Warning: Could not parse dict: {e}", file=sys.stderr)
        return {}

def _find_balanced_dict(text: str, start: int) -> int:
    """Find the end of a balanced dictionary string"""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1

# ─────────────────────────────────────────────────────────────────────────────
# Metadata Parsing (FIXED - parse latest entry, extract primitives only)
# ─────────────────────────────────────────────────────────────────────────────
def parse_cmds_txt(run_dir: Path) -> dict:
    """Parse cmds.txt and extract the LATEST training configuration"""
    meta = {
        #"run_name": run_dir.name, # default
        "run_name": f"{run_dir.parent.name}_{run_dir.name}", # spl case
        "steps_per_cycle": None,
        "episode_length": None,
        "sym_loss_coeff": None,
        "loop_phase_obs": None,
        "max_cycles": None,
        "horizon": None,
        "num_envs": None,
        "actor_lr": None,
        "critic_lr": None,
        "disc_lr": None,
        "fps": None,
        "save_interval": None,
        "log_interval": None,
    }
    
    cmds_path = run_dir / CMDS_NAME
    if not cmds_path.exists():
        raise FileNotFoundError(f"cmds.txt not found in {run_dir}")
    
    text = cmds_path.read_text(errors="replace")
    
    # Find ALL timestamp positions in the file
    timestamp_pattern = r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+'
    timestamp_matches = list(re.finditer(timestamp_pattern, text))
    
    if not timestamp_matches:
        raise ValueError(f"No timestamped entries found in {cmds_path}")
    
    # Get the last timestamp position (latest entry)
    last_timestamp_match = timestamp_matches[-1]
    last_timestamp_start = last_timestamp_match.start()
    
    # Find where this entry ends: at the next timestamp OR end of file
    # We need to find the NEXT timestamp after the current one
    next_timestamp_start = len(text)  # Default to EOF
    for i, match in enumerate(timestamp_matches):
        if match.start() > last_timestamp_start:
            next_timestamp_start = match.start()
            break
    
    # Extract the full entry text (from last timestamp to next timestamp or EOF)
    latest_entry_text = text[last_timestamp_start:next_timestamp_start].strip()
    
    # Extract run name from command
    m = re.search(r'"main\.py\s+([\w/]+\.py)', latest_entry_text)
    if m:
        #meta["run_name"] = Path(m.group(1)).stem
        meta["run_name"] = Path(m.group(1)).with_suffix("").as_posix()
    
    # Parse Training Params from latest entry
    train_params = {}
    for m_tp in re.finditer(r"Training Params:\s*(\{)", latest_entry_text):
        end = _find_balanced_dict(latest_entry_text, m_tp.start(1))
        dict_text = latest_entry_text[m_tp.start(1): end + 1]
        train_params = _safe_eval_dict(dict_text)
        break
    
    # Parse Environment Params from latest entry
    env_params = {}
    for m_ep in re.finditer(r"Environment Params:\s*(\{)", latest_entry_text):
        end = _find_balanced_dict(latest_entry_text, m_ep.start(1))
        dict_text = latest_entry_text[m_ep.start(1): end + 1]
        env_params = _safe_eval_dict(dict_text)
        break
    
    # Extract required parameters (raise exception if not found)
    for param in REQUIRED_TRAIN_PARAMS:
        if param in train_params:
            meta[param] = train_params[param]
        else:
            raise KeyError(f"Required training parameter '{param}' not found in {cmds_path}")
    
    for param in REQUIRED_ENV_PARAMS:
        if param in env_params:
            meta[param] = env_params[param]
        else:
            raise KeyError(f"Required environment parameter '{param}' not found in {cmds_path}")
    
    # Extract optional learning rates (may not exist)
    meta["actor_lr"] = train_params.get("actor_lr")
    meta["critic_lr"] = train_params.get("critic_lr")
    meta["disc_lr"] = train_params.get("disc_lr")
    
    return meta

# ─────────────────────────────────────────────────────────────────────────────
# CSV Loading (FIXED - handle multiple disc/reward columns)
# ─────────────────────────────────────────────────────────────────────────────
def load_run(run_dir: Path) -> Tuple[dict, Optional[pd.DataFrame]]:
    """Load run data with proper handling of multiple discriminator/reward columns"""
    meta = parse_cmds_txt(run_dir)
    csv_path = run_dir / CSV_NAME
    
    if not csv_path.exists():
        print(f"  [!] No {CSV_NAME} in {run_dir}", file=sys.stderr)
        return meta, None
    
    df = pd.read_csv(csv_path)
    # Filter entries to keep only log_interval aligned epochs
    if "epoch" in df.columns and meta.get("log_interval"):
        log_interval = meta["log_interval"]
        df = df[df["epoch"] % log_interval == 0].reset_index(drop=True)
    
    # Identify all discriminator columns
    real_cols = [c for c in df.columns if c.startswith("score_real_")]
    fake_cols = [c for c in df.columns if c.startswith("score_fake_")]
    disc_reward_cols = [c for c in df.columns if c.startswith("disc_reward_")]
    
    # Store individual disc column names for separate plots
    meta["disc_column_names"] = []
    for col in real_cols:
        name = col.replace('score_real_', '')
        if name not in meta["disc_column_names"]:
            meta["disc_column_names"].append(name)
    
    # Create aggregated columns for summary
    if real_cols:
        df["score_real"] = df[real_cols].mean(axis=1)
    if fake_cols:
        df["score_fake"] = df[fake_cols].mean(axis=1)
    if real_cols and fake_cols:
        df["disc_gap"] = df["score_real"] - df["score_fake"]
    if disc_reward_cols:
        df["disc_reward"] = df[disc_reward_cols].mean(axis=1)
    
    # Handle symmetry loss
    coeff = meta["sym_loss_coeff"]
    if "sym_loss" in df.columns:
        if coeff > 0:
            df["sym_loss_norm"] = df["sym_loss"] / coeff
        else:
            df["sym_loss_norm"] = df["sym_loss"]
    
    # Calculate lifetime in cycles
    spc = meta["steps_per_cycle"]
    if spc and spc > 0 and "lifetime" in df.columns:
        df["lifetime_cycles"] = df["lifetime"] / spc
    else:
        df["lifetime_cycles"] = df.get("lifetime", np.nan)
    
    # Identify all reward columns (containing 'reward' but not disc_reward or reward_mean)
    reward_cols = [c for c in df.columns if 'reward' in c.lower() 
                   and not c.startswith('disc_reward') 
                   and c != 'reward_mean'
                   and c != 'terminate_reward']
    meta["reward_columns"] = reward_cols
    
    # Get actual last epoch from CSV (not just row count)
    if "epoch" in df.columns:
        meta["last_epoch"] = int(df["epoch"].iloc[-1])
    else:
        meta["last_epoch"] = len(df) * meta["log_interval"]
    
    meta["num_log_entries"] = len(df)
    
    return meta, df

# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def smooth(y: np.ndarray, w: int = 15) -> np.ndarray:
    """Smooth array with moving average"""
    arr = np.asarray(y, dtype=float)
    n = arr.size
    if n < 3:
        return arr
    # Keep odd window size for symmetric smoothing.
    w = max(3, int(w))
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w < 3:
        return arr
    valid = np.isfinite(arr)
    if valid.sum() < 2:
        return arr
    if not valid.all():
        arr = np.interp(np.arange(n), np.flatnonzero(valid), arr[valid])
    kernel = np.ones(w, dtype=float) / float(w)
    pad = w // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(arr_pad, kernel, mode="valid")


def _shorten_run_names(names: List[str]) -> Tuple[List[str], str]:
    """Strip shared prefix from run names while keeping labels unique."""
    if not names:
        return [], ""
    cleaned = [str(n).replace("\\", "/").strip() for n in names]
    if len(cleaned) == 1:
        return cleaned, ""

    split_names = [n.split("/") for n in cleaned]
    min_parts = min(len(parts) for parts in split_names)
    shared_parts = 0
    for idx in range(min_parts):
        token = split_names[0][idx]
        if all(parts[idx] == token for parts in split_names):
            shared_parts += 1
        else:
            break
    if shared_parts > 0:
        prefix = "/".join(split_names[0][:shared_parts])
        shortened = ["/".join(parts[shared_parts:]) or parts[-1] for parts in split_names]
    else:
        prefix = os.path.commonprefix(cleaned)
        cut = len(prefix)
        while cut > 0 and prefix[cut - 1] not in ("/", "_", "-", "."):
            cut -= 1
        prefix = prefix[:cut]
        shortened = [n[cut:] if cut > 0 else n for n in cleaned]
        shortened = [s.lstrip("/_.- ") or n for s, n in zip(shortened, cleaned)]

    # Ensure uniqueness if stripping created collisions.
    counts: Dict[str, int] = {}
    unique_names: List[str] = []
    for idx, s in enumerate(shortened):
        base = s or cleaned[idx]
        counts[base] = counts.get(base, 0) + 1
        unique_names.append(base if counts[base] == 1 else f"{base}#{counts[base]}")
    return unique_names, prefix

def calculate_trend(values: np.ndarray) -> float:
    """Calculate linear trend (slope) of values"""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])

def period_stats(series: pd.Series, start_idx: int, end_idx: int) -> dict:
    """Calculate statistics for a specific period"""
    chunk = series.iloc[start_idx:end_idx].dropna()
    if len(chunk) == 0:
        return {"mean": np.nan, "std": np.nan, "trend_1st": np.nan, "trend_2nd": np.nan, "acceleration": np.nan}
    
    n = len(chunk)
    half = n // 2
    
    mean_val = float(chunk.mean())
    std_val = float(chunk.std())
    
    if half >= 2:
        trend_1st = calculate_trend(chunk.iloc[:half].values)
        trend_2nd = calculate_trend(chunk.iloc[half:].values)
    else:
        trend_1st = calculate_trend(chunk.values)
        trend_2nd = trend_1st
    
    return {
        "mean": mean_val,
        "std": std_val,
        "trend_1st": trend_1st,
        "trend_2nd": trend_2nd,
        "acceleration": trend_2nd - trend_1st
    }

# ─────────────────────────────────────────────────────────────────────────────
# Tables (RESTRUCTURED - runs as columns, metrics as rows)
# ─────────────────────────────────────────────────────────────────────────────
def get_lr_params(meta: dict) -> str:
    """Get learning rate parameters as formatted string"""
    lrs = []
    if meta.get("actor_lr") is not None:
        lrs.append(f"actor={meta['actor_lr']}")
    if meta.get("critic_lr") is not None:
        lrs.append(f"critic={meta['critic_lr']}")
    if meta.get("disc_lr") is not None:
        lrs.append(f"disc={meta['disc_lr']}")
    return ", ".join(lrs) if lrs else "N/A"

PARAM_FIELDS = [
    ("run_name", "Run Name"),
    ("steps_per_cycle", "Steps"),
    ("episode_length", "Episode Length"),
    ("max_cycles", "Max Cycles"),
    ("sym_loss_coeff", "Sym Loss Coeff"),
    ("loop_phase_obs", "Phase Obs"),
    ("horizon", "Horizon"),
    ("num_envs", "Num Envs"),
    ("lr_params", "Learning Rates"),
    ("fps", "FPS"),
    ("save_interval", "Save Interval"),
    ("log_interval", "Log Interval"),
]

def build_param_table(runs: List[Tuple[dict, pd.DataFrame]]) -> pd.DataFrame:
    """Build parameter comparison table"""
    rows = []
    for meta, df in runs:
        row = {label: meta.get(k, "N/A") for k, label in PARAM_FIELDS if k != "lr_params"}
        row["Learning Rates"] = get_lr_params(meta)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.index = [m["run_name"] for m, _ in runs]
    return df

def highlight_diffs(param_df: pd.DataFrame) -> set:
    """Find parameters that differ across runs"""
    return {
        col for col in param_df.columns
        if col != "Run Name" and len(set(str(v) for v in param_df[col])) > 1
    }

def build_period_tables(runs: List[Tuple[dict, pd.DataFrame]], 
                        save_interval: int,
                        log_interval: int,
                        metrics: List[str]) -> List[pd.DataFrame]:
    """
    Build tables for each save_interval period.
    Returns list of DataFrames, one per period.
    Format: mean ±std | t1=1st_half; t2=2nd_half
    """
    # Find minimum epochs across all runs (use actual epoch values)
    min_epochs = min(m["last_epoch"] for m, _ in runs)
    
    # Calculate number of complete periods
    rows_per_period = save_interval // log_interval
    n_periods = min_epochs // save_interval
    
    if n_periods == 0:
        return []
    
    tables = []
    for period_idx in range(n_periods):
        start_epoch = period_idx * save_interval
        end_epoch = (period_idx + 1) * save_interval
        
        # Convert epochs to row indices
        start_row = start_epoch // log_interval
        end_row = end_epoch // log_interval
        
        # Build table for this period (runs as columns, metrics as rows)
        period_data = {"Metric": []}
        
        for meta, df in runs:
            period_data[meta["run_name"]] = []
        
        for metric in metrics:
            period_data["Metric"].append(metric)
            
            for meta, df in runs:
                if df is None or metric not in df.columns:
                    period_data[meta["run_name"]].append("N/A")
                    continue
                
                stats = period_stats(df[metric], start_row, end_row)
                
                # Format without newlines - use pipe separator
                mean_str = f"{stats['mean']:.4f}"
                std_str = f"±{stats['std']:.4f}"
                trend_str = f"t1={stats['trend_1st']:.2e}; t2={stats['trend_2nd']:.2e}"
                
                # Add coef for sym_loss
                if metric == "sym_loss_norm":
                    coef = meta.get("sym_loss_coeff", 0)
                    period_data[meta["run_name"]].append(f"{mean_str} {std_str} | {trend_str} (coef={coef})")
                else:
                    period_data[meta["run_name"]].append(f"{mean_str} {std_str} | {trend_str}")
        
        lengths = {k: len(v) for k, v in period_data.items()}
        if len(set(lengths.values())) != 1:
            print(
                "[DEBUG] build_period_tables length mismatch: "
                f"period_idx={period_idx}, epochs={start_epoch}-{end_epoch}, lengths={lengths}"
            )

        df_period = pd.DataFrame(period_data).set_index("Metric")
        df_period.columns.name = f"Period {period_idx + 1} (Epochs {start_epoch}-{end_epoch})"
        tables.append(df_period)
    
    return tables

def build_acceleration_table(runs: List[Tuple[dict, pd.DataFrame]],
                             save_interval: int,
                             log_interval: int,
                             metrics: List[str]) -> pd.DataFrame:
    """
    Build acceleration table showing trend changes across periods.
    """
    min_epochs = min(m["last_epoch"] for m, _ in runs)
    rows_per_period = save_interval // log_interval
    n_periods = min_epochs // save_interval
    
    if n_periods < 2:
        return pd.DataFrame()
    
    accel_data = {"Metric": []}
    for meta, df in runs:
        accel_data[meta["run_name"]] = []
    
    for metric in metrics:
        accel_data["Metric"].append(metric)
        
        for meta, df in runs:
            if df is None or metric not in df.columns:
                accel_data[meta["run_name"]].append("N/A")
                continue
            
            accelerations = []
            for period_idx in range(n_periods - 1):
                start1 = period_idx * rows_per_period
                end1 = (period_idx + 1) * rows_per_period
                start2 = (period_idx + 1) * rows_per_period
                end2 = (period_idx + 2) * rows_per_period
                
                stats1 = period_stats(df[metric], start1, end1)
                stats2 = period_stats(df[metric], start2, end2)
                
                if not np.isnan(stats1.get("acceleration", np.nan)) and \
                   not np.isnan(stats2.get("acceleration", np.nan)):
                    accelerations.append(stats2["acceleration"] - stats1["acceleration"])
            
            if accelerations:
                avg_accel = np.mean(accelerations)
                accel_data[meta["run_name"]].append(f"{avg_accel:.4e}")
            else:
                accel_data[meta["run_name"]].append("N/A")
    
    df_accel = pd.DataFrame(accel_data).set_index("Metric")
    df_accel.columns.name = "Overall Acceleration (trend change per period)"
    return df_accel

# ─────────────────────────────────────────────────────────────────────────────
# Plotting (UPDATED - compact with proper ratios & labels)
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(runs: List[Tuple[dict, pd.DataFrame]], 
                    output_path: str,
                    save_interval: int,
                    log_interval: int,
                    common_base_name: str):
    """
    Generate comparison plots with trend insets beside each metric.
    Layout: n_rows × 2 columns (main:trend = 3:1 width ratio)
    """
    
    # Determine which metrics to plot
    has_sym = any(
        df is not None and "sym_loss_norm" in df.columns
        and df["sym_loss_norm"].abs().sum() > 0
        and m.get("sym_loss_coeff", 0) > 0
        for m, df in runs
    )
    
    # Check for multiple discriminator columns
    all_disc_names = set()
    for meta, df in runs:
        if meta.get("disc_column_names"):
            all_disc_names.update(meta["disc_column_names"])
    
    # Check for multiple reward columns
    all_reward_cols = set()
    for meta, df in runs:
        if meta.get("reward_columns"):
            all_reward_cols.update(meta["reward_columns"])
    
    # Calculate minimum epochs across runs
    min_epochs = min(m["last_epoch"] for m, _ in runs)
    
    # Build save intervals list
    save_intervals_list = list(range(0, min_epochs + save_interval, save_interval))
    n_periods = len(save_intervals_list) - 1
    
    # Build metric list with titles
    metrics_to_plot = [
        ("lifetime_cycles", "Lifetime (cycles)", "cycles", False),
        ("score_real", "Real Scores", "score", False),
        ("score_fake", "Fake Scores", "score", False),
        ("disc_gap", "Discriminator Gap", "gap", False),
        ("value_loss", "Value Loss", "loss", True),
        ("policy_loss", "Policy Loss", "loss", False),
        ("reward_mean", "Reward Mean", "reward", False),
    ]
    
    # Add symmetry loss if applicable
    if has_sym:
        metrics_to_plot.append(("sym_loss_norm", "Symmetry Loss (normalized)", "loss", False))
    
    # Add additional reward columns
    for reward_col in sorted(all_reward_cols):
        display_name = reward_col.replace('reward_', '').replace('_', ' ').title()
        metrics_to_plot.append((reward_col, f"Reward: {display_name}", "reward", False))
    
    # Per-head plots only when some run logs multiple discriminators. Union of
    # single-disc suffixes across runs (e.g. walk/full vs run/full) is not multi-disc.
    any_run_has_multiple_discs = any(
        len(meta.get("disc_column_names") or []) > 1 for meta, _ in runs
    )
    if any_run_has_multiple_discs and all_disc_names:
        for disc_name in sorted(all_disc_names):
            metrics_to_plot.append((f"score_real_{disc_name}", f"Real ({disc_name})", "score", False))
            metrics_to_plot.append((f"score_fake_{disc_name}", f"Fake ({disc_name})", "score", False))
    
    n_metrics = len(metrics_to_plot)
    
    # Layout: n_rows × 2 columns with width ratios 3:1
    n_cols = 2
    n_rows = n_metrics
    
    fig_height = n_rows * 4
    fig = plt.figure(figsize=(20, fig_height))
    
    # Calculate a fixed ~0.8 inch top margin so it doesn't scale infinitely with rows
    top_margin = 1.0 - (0.8 / fig_height)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, 
                          hspace=0.35, wspace=0.15,
                          width_ratios=[3, 1], top=top_margin)
    
    # Legend handles with updated styling
    legend_handles = [
        Line2D([0], [0], color=COLORS[i % len(COLORS)], 
            lw=LINE_CONFIG["smooth_lw"], alpha=LINE_CONFIG["smooth_alpha"],
            label=meta["run_name"])
        for i, (meta, _) in enumerate(runs)
    ]
    
    # Plot title with common base name
    if common_base_name:
        plot_title = f"{common_base_name.replace('_', ' ').title()} Training Comparison"
    else:
        plot_title = "Training Run Comparison"
    
    def _plot_main(ax, df_col, title, ylabel, runs_data, save_intervals, log=False):
        """Plot main chart with save interval lines and min/max connections"""
        
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=GRID_CONFIG["alpha"], linewidth=GRID_CONFIG["linewidth"])
        
        # Add vertical lines at save intervals
        for si in save_intervals:
            if si < min_epochs:
                ax.axvline(x=si, color='gray', lw=GRID_CONFIG["save_interval_lw"], 
                        ls='--', alpha=GRID_CONFIG["save_interval_alpha"])
        
        # Collect min/max points per period for each run
        all_min_points = {i: [] for i in range(len(runs_data))}
        all_max_points = {i: [] for i in range(len(runs_data))}
        
        # Plot each run
        for i, (meta, df) in enumerate(runs_data):
            if df is None or df_col not in df.columns:
                continue
            
            c = COLORS[i % len(COLORS)]
            x = df["epoch"].values if "epoch" in df.columns else np.arange(len(df)) * log_interval
            y = df[df_col].values.astype(float)
            
            # Limit to min_epochs
            mask = x <= min_epochs
            x, y = x[mask], y[mask]
            
            # Main plot line (updated styling)
            ax.plot(x, y, color=c, alpha=LINE_CONFIG["raw_alpha"], lw=LINE_CONFIG["raw_lw"])
            smoothed_y = smooth(y, w=21)
            ax.plot(x, smoothed_y, color=c, lw=LINE_CONFIG["smooth_lw"], 
                alpha=LINE_CONFIG["smooth_alpha"], label=meta["run_name"])
            
            # Find min/max in each period
            for si_idx in range(len(save_intervals) - 1):
                si = save_intervals[si_idx]
                end_si = save_intervals[si_idx + 1]
                period_mask = (x >= si) & (x < end_si)
                if period_mask.sum() > 0:
                    period_y = y[period_mask]
                    period_x = x[period_mask]
                    min_idx = np.argmin(period_y)
                    max_idx = np.argmax(period_y)
                    
                    min_pt = (period_x[min_idx], period_y[min_idx])
                    max_pt = (period_x[max_idx], period_y[max_idx])
                    
                    all_min_points[i].append(min_pt)
                    all_max_points[i].append(max_pt)
        
        # Draw min/max connection lines (very subtle, grid-like texture)
        for i in range(len(runs_data)):
            c = COLORS[i % len(COLORS)]
            
            # Connect minimum points across periods
            if len(all_min_points[i]) > 1:
                min_x = [pt[0] for pt in all_min_points[i]]
                min_y = [pt[1] for pt in all_min_points[i]]
                ax.plot(min_x, min_y, color=c, lw=LINE_CONFIG["connect_lw"], 
                    ls='-', alpha=LINE_CONFIG["connect_alpha"], zorder=3)
            
            # Connect maximum points across periods
            if len(all_max_points[i]) > 1:
                max_x = [pt[0] for pt in all_max_points[i]]
                max_y = [pt[1] for pt in all_max_points[i]]
                ax.plot(max_x, max_y, color=c, lw=LINE_CONFIG["connect_lw"], 
                    ls='-', alpha=LINE_CONFIG["connect_alpha"], zorder=3)
        
        # Plot min/max markers with labels (smaller, cleaner symbols)
        for i, (meta, df) in enumerate(runs_data):
            if df is None or df_col not in df.columns:
                continue
            
            c = COLORS[i % len(COLORS)]
            
            for si_idx, (min_pt, max_pt) in enumerate(zip(all_min_points[i], all_max_points[i])):
                # Minimum marker
                ax.scatter(min_pt[0], min_pt[1], color=c, s=LINE_CONFIG["marker_size"],
                        marker='v', alpha=LINE_CONFIG["marker_alpha"], zorder=5, 
                        edgecolors='white', linewidths=0.22)
                # Maximum marker
                ax.scatter(max_pt[0], max_pt[1], color=c, s=LINE_CONFIG["marker_size"],
                        marker='^', alpha=LINE_CONFIG["marker_alpha"], zorder=5, 
                        edgecolors='white', linewidths=0.22)
                
                # Add text labels only for last period (avoid clutter)
                if si_idx == len(all_min_points[i]) - 1 and len(all_min_points[i]) > 1:
                    ax.text(min_pt[0], min_pt[1], f'  ({min_pt[0]:.0f}, {min_pt[1]:.3f})',
                        fontsize=5, color=c, alpha=0.75, va='bottom')
                    ax.text(max_pt[0], max_pt[1], f'  ({max_pt[0]:.0f}, {max_pt[1]:.3f})',
                        fontsize=5, color=c, alpha=0.75, va='bottom')

    def _plot_trend(ax, df_col, runs_data, save_intervals, metric_name):
        """Plot trend chart with acceleration values"""
        
        ax.set_title("Trend", fontsize=8, fontweight="bold")
        ax.set_xlabel("Period")
        ax.set_ylabel("Slope")
        ax.grid(True, alpha=GRID_CONFIG["alpha"], linewidth=GRID_CONFIG["linewidth"])
        ax.tick_params(labelsize=7)
        ax.set_facecolor('#FAFAFA')  # Slight off-white background for contrast
        
        # Calculate trends for each period
        for i, (meta, df) in enumerate(runs_data):
            if df is None or df_col not in df.columns:
                continue
            
            c = COLORS[i % len(COLORS)]
            y = df[df_col].values.astype(float)
            x = df["epoch"].values if "epoch" in df.columns else np.arange(len(df)) * log_interval
            
            mask = x <= min_epochs
            x, y = x[mask], y[mask]
            
            trends = []
            for si_idx in range(n_periods):
                start = save_intervals[si_idx]
                end = save_intervals[si_idx + 1]
                period_mask = (x >= start) & (x < end)
                if period_mask.sum() > 1:
                    trend = calculate_trend(smooth(y[period_mask], w=7))
                    trends.append(trend)
            
            if trends:
                ax.plot(range(1,len(trends)+1), trends, color=c, lw=0.95, marker='o',
                    markersize=2.6, alpha=0.85, label=meta["run_name"][:10])
        
        # Add acceleration summary as text box (outside plot area)
        if n_periods >= 2:
            accel_text = []
            for i, (meta, df) in enumerate(runs_data):
                if df is None or df_col not in df.columns:
                    continue
                
                y = df[df_col].values.astype(float)
                x = df["epoch"].values if "epoch" in df.columns else np.arange(len(df)) * log_interval
                
                mask = x <= min_epochs
                x, y = x[mask], y[mask]
                
                accelerations = []
                for si_idx in range(n_periods - 1):
                    mask1 = (x >= save_intervals[si_idx]) & (x < save_intervals[si_idx+1])
                    mask2 = (x >= save_intervals[si_idx+1]) & (x < save_intervals[si_idx+2])
                    if mask1.sum() > 1 and mask2.sum() > 1:
                        t1 = calculate_trend(smooth(y[mask1], w=7))
                        t2 = calculate_trend(smooth(y[mask2], w=7))
                        accelerations.append(t2 - t1)
                
                if accelerations:
                    avg_accel = np.mean(accelerations)
                    accel_text.append(f"{meta['run_name'][:8]}: {avg_accel:+.2e}")
            
            # Place acceleration legend below x-axis
            if accel_text:
                ax.text(.99, 0.02, "Avg Acceleration:\n" + "\n".join(accel_text),
                       transform=ax.transAxes, fontsize=6, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                current_ymin, current_ymax = ax.get_ylim()
                # Lower the bottom limit by 10-20% of the total range
                padding = (current_ymax - current_ymin) * 0.15 
                ax.set_ylim(current_ymin - padding, current_ymax)
                
   
    
    # Create plots for each metric
    for plot_idx, (df_col, title, ylabel, use_log) in enumerate(metrics_to_plot):
        # Main plot (column 0, wider)
        ax_main = fig.add_subplot(gs[plot_idx, 0])
        _plot_main(ax_main, df_col, title, ylabel, runs, save_intervals_list, log=use_log)
        
        # Trend plot (column 1, 1:1 ratio)
        ax_trend = fig.add_subplot(gs[plot_idx, 1])
        _plot_trend(ax_trend, df_col, runs, save_intervals_list, df_col)
    
    # Legend at top
    fig.legend(handles=legend_handles,
               loc="upper center",
               ncol=max(1, len(runs)),
               mode="expand",
               fontsize=8,
               bbox_to_anchor=(0.1, 0.95, 0.80, 0.05),
               borderaxespad=0.0,
               framealpha=0.9)
    
    fig.suptitle(plot_title, fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Chart saved → {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Conclusive Summary (using LAST period data only)
# ─────────────────────────────────────────────────────────────────────────────
def _winner_last_period(runs: List[Tuple[dict, pd.DataFrame]], col: str, higher_is_better: bool,
                        last_period_start: int, last_period_end: int, log_interval: int) -> str:
    """Find the best run for a given metric in last period"""
    best_val, best_name = None, None
    for meta, df in runs:
        if df is None or col not in df.columns:
            continue
        
        start_row = last_period_start // log_interval
        end_row = last_period_end // log_interval
        chunk = df[col].iloc[start_row:end_row].dropna()
        
        if len(chunk) == 0:
            continue
        
        v = float(chunk.mean())
        if not np.isfinite(v):
            continue
        
        if best_val is None or (higher_is_better and v > best_val) or \
           (not higher_is_better and v < best_val):
            best_val, best_name = v, meta["run_name"]
    return best_name

def conclusive_summary(runs: List[Tuple[dict, pd.DataFrame]], 
                       save_interval: int,
                       log_interval: int) -> str:
    """Generate comprehensive summary text using LAST period data only"""
    lines = ["\n", "=" * 70, "  COMPARATIVE SUMMARY", "=" * 70]
    
    # Calculate last period
    min_epochs = min(m["last_epoch"] for m, _ in runs)
    n_periods = min_epochs // save_interval
    last_period_start = (n_periods - 1) * save_interval
    last_period_end = n_periods * save_interval
    
    # Parameter differences
    param_df = build_param_table(runs)
    diff_cols = highlight_diffs(param_df)
    if diff_cols:
        lines.append("\nKey parameter differences:")
        names = [r[0]["run_name"] for r in runs]
        for col in sorted(diff_cols):
            parts = [f"{n}={v}" for n, v in zip(names, param_df[col].tolist())]
            lines.append(f"    {col:22s}  {' | '.join(parts)}")
    else:
        lines.append("\n(All runs share identical parameters)")
    
    # Performance winners (using LAST period data only)
    lines.append(f"\nPerformance winners (Period {n_periods}: Epochs {last_period_start}-{last_period_end}):")
    metrics_winners = [
        ("lifetime_cycles", True, "Longest survival (cycles)"),
        ("score_fake", True, "Highest fake score"),
        ("disc_gap", False, "Smallest disc gap"),
        ("reward_mean", True, "Highest mean reward"),
        ("value_loss", False, "Lowest value loss"),
        ("policy_loss", False, "Lowest policy loss"),
    ]
    
    # Add sym_loss only if any run has coeff > 0
    if any(m.get("sym_loss_coeff", 0) > 0 for m, _ in runs):
        metrics_winners.append(("sym_loss_norm", False, "Lowest sym error"))
    
    for col, hib, label in metrics_winners:
        if not any(df is not None and col in df.columns for _, df in runs):
            continue
        w = _winner_last_period(runs, col, hib, last_period_start, last_period_end, log_interval)
        if w:
            lines.append(f"    ✓  {label:40s}  → {w}")
    
    # Per-run assessment
    lines.append("\nPer-run assessment:")
    for meta, df in runs:
        name = meta["run_name"]
        lines.append(f"\n── {name} ──")
        
        if df is None:
            lines.append("    [!] No training data.")
            continue
        
        # Get last period stats
        start_row = last_period_start // log_interval
        end_row = last_period_end // log_interval
        
        # Lifetime
        if "lifetime_cycles" in df.columns:
            chunk = df["lifetime_cycles"].iloc[start_row:end_row].dropna()
            if len(chunk) > 0:
                mean_val = chunk.mean()
                std_val = chunk.std()
                mc = meta.get("max_cycles") or "?"
                pct_str = f"{mean_val/mc*100:.1f}%" if isinstance(mc, (int, float)) else "N/A"
                lines.append(f"    Lifetime:    {mean_val:.2f} ± {std_val:.2f} cycles  "
                            f"({pct_str} of max_cycles={mc})")
        
        # Discriminator
        if "score_fake" in df.columns:
            chunk = df["score_fake"].iloc[start_row:end_row].dropna()
            if len(chunk) > 0:
                sf_mean, sf_std = chunk.mean(), chunk.std()
                line = f"    Disc fake:   {sf_mean:.4f} ± {sf_std:.4f}"
                if "score_real" in df.columns:
                    chunk_r = df["score_real"].iloc[start_row:end_row].dropna()
                    if len(chunk_r) > 0:
                        line += f"   real: {chunk_r.mean():.4f} ± {chunk_r.std():.4f}"
                if "disc_gap" in df.columns:
                    chunk_g = df["disc_gap"].iloc[start_row:end_row].dropna()
                    if len(chunk_g) > 0:
                        line += f"   gap: {chunk_g.mean():.4f}"
                lines.append(line)
        
        # Other metrics
        for col, label in [("reward_mean", "Reward"), ("value_loss", "Value loss"),
                          ("policy_loss", "Policy loss")]:
            if col in df.columns:
                chunk = df[col].iloc[start_row:end_row].dropna()
                if len(chunk) > 0:
                    lines.append(f"    {label:12s} {chunk.mean():.4f} ± {chunk.std():.4f}")
        
        # Symmetry loss (only if coeff > 0)
        if "sym_loss_norm" in df.columns and meta.get("sym_loss_coeff", 0) > 0:
            chunk = df["sym_loss_norm"].iloc[start_row:end_row].dropna()
            if len(chunk) > 0:
                lines.append(f"    Sym loss:    {chunk.mean():.4f} ± {chunk.std():.4f}  "
                            f"(coeff={meta['sym_loss_coeff']})")
        
        # Health flags
        health_checks = [
            (lambda: "disc_gap" in df.columns and 
             len(df["disc_gap"].iloc[start_row:end_row].dropna()) > 0 and
             df["disc_gap"].iloc[start_row:end_row].mean() > 0.6,
             "⚠ discriminator dominating"),
            (lambda: "disc_gap" in df.columns and
             len(df["disc_gap"].iloc[start_row:end_row].dropna()) > 0 and
             df["disc_gap"].iloc[start_row:end_row].mean() < 0.05,
             "⚠ disc gap very small — check mode collapse"),
            (lambda: "value_loss" in df.columns and
             len(df["value_loss"].iloc[start_row:end_row].dropna()) > 0 and
             df["value_loss"].iloc[start_row:end_row].mean() > 1000,
             "⚠ value loss very high — critic may be unstable"),
        ]
        
        for cond, msg in health_checks:
            try:
                if cond():
                    lines.append(f"    {msg}")
            except Exception:
                pass
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def print_table(df: pd.DataFrame, title: str = "", out: TextIO = sys.stdout):
    """Pretty print a DataFrame"""
    if df is None or df.empty:
        return
    if title:
        print(f"\n{'─'*70}\n{title}\n{'─'*70}", file=out)
    with pd.option_context("display.max_columns", None,
                          "display.width", 250,
                          "display.max_colwidth", 100):
        print(df.to_string(), file=out)

def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple ICCGAN training runs side-by-side.")
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--output", "-o", default="case_studies/comparison.png")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    output_path = Path(args.output)
    out_dir = output_path.parent if str(output_path.parent) not in ("", ".") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = out_dir / "analysis.txt"
    
    # Load all runs
    runs = []
    for path_str in args.run_dirs:
        run_dir = Path(path_str)
        if not run_dir.is_dir():
            print(f"[!] Not a directory: {run_dir}", file=sys.stderr)
            continue
        print(f"  Loading {run_dir} ...", end=" ", flush=True)
        try:
            meta, df = load_run(run_dir)
            print(f"→ {meta['last_epoch']} epochs ({meta['num_log_entries']} log entries)  ({meta['run_name']})")
            runs.append((meta, df))
        except (FileNotFoundError, KeyError) as e:
            print(f"[!] Error: {e}", file=sys.stderr)
            continue
    
    if not runs:
        print("No valid runs found.", file=sys.stderr)
        sys.exit(1)
    
    # Normalize run labels by removing shared prefix across all runs.
    run_names = [m["run_name"] for m, _ in runs]
    normalized_names, common_base_name = _shorten_run_names(run_names)
    for (meta, _), norm_name in zip(runs, normalized_names):
        meta["run_name"] = norm_name
    if common_base_name and len(common_base_name) < 3:
        common_base_name = ""
    
    # Check save_interval and log_interval consistency
    save_intervals = [m.get("save_interval", 500) for m, _ in runs]
    log_intervals = [m.get("log_interval", 10) for m, _ in runs]
    
    if len(set(save_intervals)) > 1:
        warnings.warn(f"Save intervals differ across runs: {save_intervals}. Using average.")
        save_interval = int(np.mean(save_intervals))
    else:
        save_interval = save_intervals[0]
    save_interval *= 2 # to increase analysis period
    
    if len(set(log_intervals)) > 1:
        warnings.warn(f"Log intervals differ across runs: {log_intervals}. Using average.")
        log_interval = int(np.mean(log_intervals))
    else:
        log_interval = log_intervals[0]
    with analysis_path.open("w", encoding="utf-8") as analysis_out:
        print(f"Using save_interval = {save_interval}, log_interval = {log_interval} for period analysis", file=analysis_out)
        
        # Print parameter table
        param_table = build_param_table(runs)
        print_table(param_table, "Run Parameters", out=analysis_out)
        diff = highlight_diffs(param_table)
        if diff:
            print(f"\nDiffering params: {', '.join(sorted(diff))}", file=analysis_out)
        
        # Build metrics list for tables
        metrics_for_tables = [
            "lifetime_cycles", "reward_mean", "policy_loss", "value_loss",
            "disc_gap", "score_real", "score_fake"
        ]
        
        # Add sym_loss if any run has coeff > 0
        if any(m.get("sym_loss_coeff", 0) > 0 for m, _ in runs):
            metrics_for_tables.append("sym_loss_norm")
        
        # Add additional reward columns from all runs
        for meta, df in runs:
            if meta.get("reward_columns"):
                for col in meta["reward_columns"]:
                    if col not in metrics_for_tables:
                        metrics_for_tables.append(col)
        
        # Build and print ALL period tables
        period_tables = build_period_tables(runs, save_interval, log_interval, metrics_for_tables)
        for i, table in enumerate(period_tables):
            print_table(table, f"Period {i+1} Metrics (mean ±std | t1=1st_half; t2=2nd_half)", out=analysis_out)
        
        # Build and print acceleration table
        accel_table = build_acceleration_table(runs, save_interval, log_interval, metrics_for_tables)
        if not accel_table.empty:
            print_table(accel_table, "Overall Acceleration (trend change per period)", out=analysis_out)
        
        # Print conclusive summary (using last period data)
        print(conclusive_summary(runs, save_interval, log_interval), file=analysis_out)
    
    print(f"\n📝 Analysis report saved → {analysis_path}")
    
    # Generate plots
    if not args.no_plots and any(df is not None for _, df in runs):
        #print(f"\nGenerating comparison chart → {args.output} …")
        plot_comparison(runs, args.output, save_interval, log_interval, common_base_name)

if __name__ == "__main__":
    main()