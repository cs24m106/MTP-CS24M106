#!/usr/bin/env python3
"""
eval_analyzer.py — Analyze multiple ICCGAN training runs side-by-side.
Usage:
python eval_analyzer.py run_dir1 run_dir2 [run_dir3 ...]
python eval_analyzer.py run_dir1 run_dir2 --no-plots
python eval_analyzer.py run_dir1 run_dir2 --phase-metric lifetime_cycles
Compares runs with different training configurations (phase input,
symmetry loss, motion cycle looping) using normalized metrics.
"""
import argparse
import ast
import os
import re
import sys
from pathlib import Path
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
CSV_NAME  = "training_metrics.csv"
CMDS_NAME = "cmds.txt"
PHASE_NAMES = ["early", "mid", "late"]
COLORS = [
"#2196F3", "#F44336", "#4CAF50", "#FF9800",
"#9C27B0", "#00BCD4", "#795548", "#E91E63",
]
# ─────────────────────────────────────────────────────────────────────────────
# Metadata parsing
# ─────────────────────────────────────────────────────────────────────────────
def _find_balanced_dict(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        if depth == 0:
            return i
    return len(text) - 1

def parse_cmds_txt(run_dir: Path) -> dict:
    meta = {
        "run_name":       run_dir.name,
        "path":           str(run_dir),  # Added: directory path for saving plots
        "steps_per_cycle": None,
        "episode_length":  None,
        "sym_loss_coeff":  0.0,
        "loop_phase_obs":  False,
        "max_cycles":      None,
        "horizon":         None,
        "num_envs":        None,
        "learning_rate":   None,
        "fps":             None,
        "gamma":           None,        # Added: parsed from Training Params
        "terminate_reward": None,       # Added: parsed from Training Params
    }
    cmds_path = run_dir / CMDS_NAME
    if not cmds_path.exists():
        return meta
    text = cmds_path.read_text(errors="replace")
    m = re.search(r"Run Name:\s*(.+)", text)
    if m:
        meta["run_name"] = m.group(1).strip()
    else:
        m2 = re.search(r'"main\.py\s+([\w/]+\.py)', text)
        if m2:
            meta["run_name"] = Path(m2.group(1)).stem
    train_params = {}
    env_params   = {}
    for m_tp in re.finditer(r"Training Params:\s*(\{)", text):
        end = _find_balanced_dict(text, m_tp.start(1))
        try:
            train_params = ast.literal_eval(text[m_tp.start(1): end + 1])
        except Exception:
            pass
    for m_ep in re.finditer(r"Environment Params:\s*(\{)", text):
        end = _find_balanced_dict(text, m_ep.start(1))
        try:
            env_params = ast.literal_eval(text[m_ep.start(1): end + 1])
        except Exception:
            pass
    meta["steps_per_cycle"] = env_params.get("steps_per_cycle")
    meta["episode_length"]  = env_params.get("episode_length")
    meta["fps"]             = env_params.get("fps")
    meta["max_cycles"]      = env_params.get("max_cycles")
    meta["sym_loss_coeff"]  = float(train_params.get("sym_loss_coeff", 0.0))
    meta["loop_phase_obs"]  = bool(train_params.get("loop_phase_obs", False))
    meta["horizon"]         = train_params.get("horizon")
    meta["num_envs"]        = train_params.get("num_envs")
    meta["learning_rate"]   = train_params.get("learning_rate")
    # --- ADDED: Parse gamma and terminate_reward from Training Params ---
    meta["gamma"]           = train_params.get("gamma")
    meta["terminate_reward"] = train_params.get("terminate_reward")
    return meta

# ─────────────────────────────────────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────────────────────────────────────
def load_run(run_dir: Path):
    meta = parse_cmds_txt(run_dir)
    csv_path = run_dir / CSV_NAME
    if not csv_path.exists():
        print(f"  [!] No {CSV_NAME} in {run_dir}", file=sys.stderr)
        return meta, None
    df = pd.read_csv(csv_path)
    real_cols = [c for c in df.columns if c.startswith("score_real_")]
    fake_cols = [c for c in df.columns if c.startswith("score_fake_")]
    disc_cols = [c for c in df.columns if c.startswith("disc_reward_")]
    if real_cols:
        df["score_real"] = df[real_cols].mean(axis=1)
    if fake_cols:
        df["score_fake"] = df[fake_cols].mean(axis=1)
    if real_cols and fake_cols:
        df["disc_gap"] = df["score_real"] - df["score_fake"]
    if disc_cols:
        df["disc_reward"] = df[disc_cols].mean(axis=1)
    coeff = meta["sym_loss_coeff"]
    if "sym_loss" in df.columns:
        df["sym_loss_raw"] = df["sym_loss"] / coeff if coeff > 0 else df["sym_loss"]
    spc = meta["steps_per_cycle"]
    if spc and spc > 0 and "lifetime" in df.columns:
        df["lifetime_cycles"] = df["lifetime"] / spc
    else:
        df["lifetime_cycles"] = df.get("lifetime", np.nan)
    return meta, df

# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def phase_stats(series: pd.Series) -> dict:
    n = len(series)
    if n < 6:
        return {}
    thirds = np.array_split(series.values, 3)
    result = {}
    for name, chunk in zip(PHASE_NAMES, thirds):
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk) == 0:
            continue
        x = np.arange(len(chunk), dtype=float)
        slope = float(np.polyfit(x, chunk, 1)[0]) if len(chunk) > 1 else 0.0
        result[name] = {"mean": float(np.mean(chunk)),
                        "std":  float(np.std(chunk)),
                        "slope": slope}
    if "early" in result and "late" in result:
        result["acceleration"] = result["late"]["slope"] - result["early"]["slope"]
    return result

def final_stats(series: pd.Series, window_frac: float = 0.2) -> dict:
    n = len(series)
    w = max(1, int(n * window_frac))
    tail = series.iloc[-w:].dropna()
    if len(tail) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(tail.mean()), "std": float(tail.std())}

# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────
PARAM_FIELDS = [
    ("run_name",        "Run Name"),
    ("steps_per_cycle", "Steps/Cycle"),
    ("episode_length",  "Episode Length"),
    ("max_cycles",      "Max Cycles"),
    ("sym_loss_coeff",  "Sym Loss Coeff"),
    ("loop_phase_obs",  "Phase Obs"),
    ("horizon",         "Horizon"),
    ("num_envs",        "Num Envs"),
    ("learning_rate",   "LR"),
    ("fps",             "FPS"),
    ("gamma",           "Gamma"),            # Added
    ("terminate_reward", "Term Reward"),     # Added
]

def build_param_table(runs):
    rows = [{label: m.get(k, "N/A") for k, label in PARAM_FIELDS} for m, _ in runs]
    df = pd.DataFrame(rows)
    df.index = [r[0]["run_name"] for r in runs]
    return df

def highlight_diffs(param_df: pd.DataFrame) -> set:
    return {
        col for col in param_df.columns
        if col != "Run Name" and len(set(str(v) for v in param_df[col])) > 1
    }

def build_metric_table(runs):
    rows = []
    for meta, df in runs:
        if df is None:
            continue
        row = {"Run": meta["run_name"]}
        for col, label in [
            ("lifetime_cycles", "Lifetime (cycles)"),
            ("reward_mean",     "Reward"),
            ("policy_loss",     "Policy Loss"),
            ("value_loss",      "Value Loss"),
            ("disc_gap",        "Disc Gap"),
            ("score_real",      "Score Real"),
            ("score_fake",      "Score Fake"),
        ]:
            if col in df.columns:
                s = final_stats(df[col])
                row[label] = f"{s['mean']:.4f} ± {s['std']:.4f}"
            else:
                row[label] = "N/A"
        if "sym_loss_raw" in df.columns:
            s = final_stats(df["sym_loss_raw"])
            coeff = meta["sym_loss_coeff"]
            tag = f"÷{coeff}" if coeff > 0 else "coeff=0"
            row[f"Sym Loss ({tag})"] = f"{s['mean']:.4f} ± {s['std']:.4f}"
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Run")

def build_phase_table(runs, col: str):
    rows = []
    for meta, df in runs:
        if df is None or col not in df.columns:
            continue
        ps = phase_stats(df[col])
        row = {"Run": meta["run_name"]}
        for phase in PHASE_NAMES:
            if phase in ps:
                p = ps[phase]
                row[f"{phase.capitalize()} mean"] = f"{p['mean']:.4f}"
                row[f"{phase.capitalize()} slope"] = f"{p['slope']:.4e}"
        if "acceleration" in ps:
            row["Acceleration"] = f"{ps['acceleration']:.4e}"
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Run")

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def smooth(y: np.ndarray, w: int = 15) -> np.ndarray:
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")

def plot_single_run(meta, df, output_path: str):
    """
    Generate diagnostic plot for a single run and save to its directory.
    """
    if df is None or len(df) == 0:
        print(f"  [!] No data to plot for {meta['run_name']}")
        return
    
    has_sym = "sym_loss_raw" in df.columns and df["sym_loss_raw"].abs().sum() > 0
    n_rows = 4 + (1 if has_sym else 0)
    fig = plt.figure(figsize=(18, n_rows * 4))
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.55, wspace=0.35)
    
    def _ax(row, col, span=1):
        return fig.add_subplot(gs[row, col] if span == 1 else gs[row, col:col + span])
    
    def _plot(ax, col, title, ylabel="", log=False, zoom=None, hline=None):
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        if hline is not None:
            ax.axhline(hline, color="gray", lw=0.8, ls="--", alpha=0.5)
        if df is None or col not in df.columns:
            return
        x = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
        y = df[col].values.astype(float)
        if zoom is not None:
            s = int(len(x) * (1 - zoom))
            x, y = x[s:], y[s:]
        ax.plot(x, y, color=COLORS[0], alpha=0.15, lw=0.8)
        ax.plot(x, smooth(y), color=COLORS[0], lw=2)
        ax.grid(True, alpha=0.3)
    
    # Row 0: Lifetime
    _plot(_ax(0, 0, 2), "lifetime_cycles", "Lifetime — full (cycles)",    "cycles")
    _plot(_ax(0, 2),    "lifetime_cycles", "Lifetime — last 30% (cycles)","cycles", zoom=0.3)
    mc = meta.get("max_cycles")
    if mc:
        for ax in [fig.axes[-2], fig.axes[-1]]:
            ax.axhline(mc, color=COLORS[0], lw=1, ls=":", alpha=0.6)
    
    # Row 1: Disc scores
    ax_disc = _ax(1, 0, 2)
    ax_fake = _ax(1, 2)
    if df is not None:
        c = COLORS[0]
        x = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
        if "score_real" in df.columns:
            ax_disc.plot(x, smooth(df["score_real"].values.astype(float)), color=c, lw=2)
        if "score_fake" in df.columns:
            yf = df["score_fake"].values.astype(float)
            ax_disc.plot(x, smooth(yf), color=c, lw=1.5, ls="--", alpha=0.7)
            ax_fake.plot(x, smooth(yf), color=c, lw=2)
    for ax, title in [(ax_disc, "Disc Scores: real(—) fake(--)"),
                      (ax_fake, "Fake Score (agent quality)")]:
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    # Row 2: Disc gap + reward
    _plot(_ax(2, 0),    "disc_gap",    "Disc Gap (real − fake)",    "gap",    hline=0.0)
    _plot(_ax(2, 1),    "disc_reward", "Disc Reward — full",        "reward")
    _plot(_ax(2, 2),    "disc_reward", "Disc Reward — last 30%",    "reward", zoom=0.3)
    
    # Row 3: Losses + reward
    _plot(_ax(3, 0), "value_loss",  "Value Loss (log)",  "loss", log=True)
    _plot(_ax(3, 1), "policy_loss", "Policy Loss",       "loss")
    _plot(_ax(3, 2), "reward_mean", "Reward Mean",       "reward")
    
    # Row 4 (optional): Symmetry
    if has_sym:
        _plot(_ax(4, 0, 2), "sym_loss_raw", "Symmetry Loss Raw — full",     "raw loss")
        _plot(_ax(4, 2),    "sym_loss_raw", "Symmetry Loss Raw — last 30%", "raw loss", zoom=0.3)
        notes = f"{meta['run_name']}:coeff={meta['sym_loss_coeff']}"
        fig.axes[-2].text(0.01, 0.97, notes, transform=fig.axes[-2].transAxes,
                          fontsize=7, va="top", color="gray")
    
    fig.suptitle(f"ICCGAN Training Diagnostics — {meta['run_name']}", fontsize=14,
                 fontweight="bold", y=1.03)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Chart saved → {output_path}")

def plot_comparison(runs, output_path: str):
    """
    Generate consolidated comparison plot for multiple runs (optional).
    """
    has_sym = any(
        df is not None and "sym_loss_raw" in df.columns
        and df["sym_loss_raw"].abs().sum() > 0
        for _, df in runs
    )
    n_rows = 4 + (1 if has_sym else 0)
    fig = plt.figure(figsize=(18, n_rows * 4))
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.55, wspace=0.35)
    legend_handles = [
        Line2D([0], [0], color=COLORS[i % len(COLORS)], lw=2,
               label=meta["run_name"])
        for i, (meta, _) in enumerate(runs)
    ]
    
    def _ax(row, col, span=1):
        return fig.add_subplot(gs[row, col] if span == 1 else gs[row, col:col + span])
    
    def _plot(ax, col, title, ylabel="", log=False, zoom=None, hline=None):
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        if hline is not None:
            ax.axhline(hline, color="gray", lw=0.8, ls="--", alpha=0.5)
        for i, (meta, df) in enumerate(runs):
            if df is None or col not in df.columns:
                continue
            x = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
            y = df[col].values.astype(float)
            if zoom is not None:
                s = int(len(x) * (1 - zoom))
                x, y = x[s:], y[s:]
            c = COLORS[i % len(COLORS)]
            ax.plot(x, y, color=c, alpha=0.15, lw=0.8)
            ax.plot(x, smooth(y), color=c, lw=2)
            ax.grid(True, alpha=0.3)
    
    # Row 0: Lifetime
    _plot(_ax(0, 0, 2), "lifetime_cycles", "Lifetime — full (cycles)",    "cycles")
    _plot(_ax(0, 2),    "lifetime_cycles", "Lifetime — last 30% (cycles)","cycles", zoom=0.3)
    for i, (meta, _) in enumerate(runs):
        mc = meta.get("max_cycles")
        if mc:
            for ax in [fig.axes[-2], fig.axes[-1]]:
                ax.axhline(mc, color=COLORS[i % len(COLORS)], lw=1, ls=":", alpha=0.6)
    
    # Row 1: Disc scores
    ax_disc = _ax(1, 0, 2)
    ax_fake = _ax(1, 2)
    for i, (meta, df) in enumerate(runs):
        if df is None:
            continue
        c = COLORS[i % len(COLORS)]
        x = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
        if "score_real" in df.columns:
            ax_disc.plot(x, smooth(df["score_real"].values.astype(float)), color=c, lw=2)
        if "score_fake" in df.columns:
            yf = df["score_fake"].values.astype(float)
            ax_disc.plot(x, smooth(yf), color=c, lw=1.5, ls="--", alpha=0.7)
            ax_fake.plot(x, smooth(yf), color=c, lw=2)
    for ax, title in [(ax_disc, "Disc Scores: real(—) fake(--)"),
                      (ax_fake, "Fake Score (agent quality)")]:
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    # Row 2: Disc gap + reward
    _plot(_ax(2, 0),    "disc_gap",    "Disc Gap (real − fake)",    "gap",    hline=0.0)
    _plot(_ax(2, 1),    "disc_reward", "Disc Reward — full",        "reward")
    _plot(_ax(2, 2),    "disc_reward", "Disc Reward — last 30%",    "reward", zoom=0.3)
    
    # Row 3: Losses + reward
    _plot(_ax(3, 0), "value_loss",  "Value Loss (log)",  "loss", log=True)
    _plot(_ax(3, 1), "policy_loss", "Policy Loss",       "loss")
    _plot(_ax(3, 2), "reward_mean", "Reward Mean",       "reward")
    
    # Row 4 (optional): Symmetry
    if has_sym:
        _plot(_ax(4, 0, 2), "sym_loss_raw", "Symmetry Loss Raw — full",     "raw loss")
        _plot(_ax(4, 2),    "sym_loss_raw", "Symmetry Loss Raw — last 30%", "raw loss", zoom=0.3)
        notes = " | ".join(
            f"{m['run_name']}:coeff={m['sym_loss_coeff']}" for m, _ in runs
        )
        fig.axes[-2].text(0.01, 0.97, notes, transform=fig.axes[-2].transAxes,
                          fontsize=7, va="top", color="gray")
    
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=min(len(runs), 6), fontsize=9,
               bbox_to_anchor=(0.5, 1.01), framealpha=0.9)
    fig.suptitle("ICCGAN Training Run Comparison", fontsize=14,
                 fontweight="bold", y=1.03)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Comparison chart saved → {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Conclusive summary
# ─────────────────────────────────────────────────────────────────────────────
def _winner(runs, col: str, higher_is_better: bool):
    best_val, best_name = None, None
    for meta, df in runs:
        if df is None or col not in df.columns:
            continue
        v = final_stats(df[col])["mean"]
        if not np.isfinite(v):
            continue
        if best_val is None or (higher_is_better and v > best_val) or \
           (not higher_is_better and v < best_val):
            best_val, best_name = v, meta["run_name"]
    return best_name

def conclusive_summary(runs) -> str:
    lines = ["=" * 70, "  COMPARATIVE SUMMARY", "=" * 70]
    param_df  = build_param_table(runs)
    diff_cols = highlight_diffs(param_df)
    if diff_cols:
        lines.append("\n  Key parameter differences:")
        names = [r[0]["run_name"] for r in runs]
        for col in sorted(diff_cols):
            parts = [f"{n}={v}" for n, v in zip(names, param_df[col].tolist())]
            lines.append(f"    {col:22s}  {' | '.join(parts)}")
    else:
        lines.append("\n  (All runs share identical parameters)")
    lines.append("")
    lines.append("  Performance winners (final 20%):")
    for col, hib, label in [
        ("lifetime_cycles", True,  "Longest survival (cycles)"),
        ("score_fake",      True,  "Highest fake score"),
        ("disc_gap",        False, "Smallest disc gap"),
        ("reward_mean",     True,  "Highest mean reward"),
        ("value_loss",      False, "Lowest value loss"),
        ("policy_loss",     False, "Lowest policy loss"),
        ("sym_loss_raw",    False, "Lowest raw symmetry error"),
    ]:
        if not any(df is not None and col in df.columns for _, df in runs):
            continue
        w = _winner(runs, col, hib)
        if w:
            lines.append(f"    ✓  {label:40s}  → {w}")
    lines.append("")
    lines.append("  Per-run assessment:")
    for meta, df in runs:
        name = meta["run_name"]
        lines.append(f"\n  ── {name} ──")
        if df is None:
            lines.append("    [!] No training data."); continue
        if "lifetime_cycles" in df.columns:
            s  = final_stats(df["lifetime_cycles"])
            mc = meta.get("max_cycles") or "?"
            pct_str = f"{s['mean']/mc*100:.1f}%" if isinstance(mc, (int,float)) else "N/A"
            lines.append(f"    Lifetime:    {s['mean']:.2f} ± {s['std']:.2f} cycles  "
                         f"({pct_str} of max_cycles={mc})")
        if "score_fake" in df.columns:
            sf  = final_stats(df["score_fake"])
            sr  = final_stats(df["score_real"])  if "score_real"  in df.columns else None
            gap = final_stats(df["disc_gap"])     if "disc_gap"    in df.columns else None
            line = f"    Disc fake:   {sf['mean']:.4f} ± {sf['std']:.4f}"
            if sr:  line += f"   real: {sr['mean']:.4f} ± {sr['std']:.4f}"
            if gap: line += f"   gap: {gap['mean']:.4f}"
            lines.append(line)
        for col, label in [("reward_mean","Reward"), ("value_loss","Value loss"),
                           ("policy_loss","Policy loss")]:
            if col in df.columns:
                s = final_stats(df[col])
                lines.append(f"    {label:12s} {s['mean']:.4f} ± {s['std']:.4f}")
        if "sym_loss_raw" in df.columns:
            s = final_stats(df["sym_loss_raw"])
            coeff = meta["sym_loss_coeff"]
            tag = f"coeff={coeff}" if coeff > 0 else "informational, coeff=0"
            lines.append(f"    Sym loss:    {s['mean']:.4f} ± {s['std']:.4f}  ({tag})")
        flags = []
        if meta.get("loop_phase_obs"):   flags.append("phase_obs=ON")
        if meta.get("sym_loss_coeff",0) > 0: flags.append(f"sym_loss={meta['sym_loss_coeff']}")
        if meta.get("steps_per_cycle"):  flags.append(f"steps/cycle={meta['steps_per_cycle']}")
        if flags: lines.append(f"    Flags:       {', '.join(flags)}")
        if "lifetime_cycles" in df.columns:
            ps = phase_stats(df["lifetime_cycles"])
            if "early" in ps and "late" in ps:
                accel = ps.get("acceleration", 0.0)
                trend = "improving" if accel > 0 else ("stable" if abs(accel) < 1e-5 else "degrading")
                lines.append(f"    Trend:       {trend}  (accel={accel:.4e})")
        # Health flags
        for cond, msg in [
            (lambda: "disc_gap" in df.columns and final_stats(df["disc_gap"])["mean"] > 0.6,
             "⚠ discriminator dominating"),
            (lambda: "disc_gap" in df.columns and final_stats(df["disc_gap"])["mean"] < 0.05,
             "⚠ disc gap very small — check mode collapse"),
            (lambda: "value_loss" in df.columns and final_stats(df["value_loss"])["mean"] > 1000,
             "⚠ value loss very high — critic may be unstable"),
            (lambda: "lifetime_cycles" in df.columns
             and isinstance(meta.get("max_cycles"), (int, float))
             and final_stats(df["lifetime_cycles"])["mean"] < meta["max_cycles"] * 0.2,
             "⚠ agent still dying early (<20% of max_cycles)"),
            (lambda: "lifetime_cycles" in df.columns
             and isinstance(meta.get("max_cycles"), (int, float))
             and final_stats(df["lifetime_cycles"])["mean"] > meta["max_cycles"] * 0.9,
             "✓ agent surviving near full episode"),
        ]:
            try:
                if cond(): lines.append(f"    {msg}")
            except Exception:
                pass
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def print_table(df: pd.DataFrame, title: str = ""):
    if df is None or df.empty:
        return
    if title:
        print(f"\n{'─'*70}\n{title}\n{'─'*70}")
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.max_colwidth", 30):
        print(df.to_string())

def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple ICCGAN training runs side-by-side.")
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    # --- REMOVED: --output argument (individual plots saved per run) ---
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating diagnostic plots")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Skip consolidated comparison plot")
    parser.add_argument("--phase-metric", default="lifetime_cycles",
                        help="Metric for phase analysis (default: lifetime_cycles)")
    args = parser.parse_args()
    
    runs = []
    for path_str in args.run_dirs:
        run_dir = Path(path_str)
        if not run_dir.is_dir():
            print(f"[!] Not a directory: {run_dir}", file=sys.stderr); continue
        print(f"  Loading {run_dir} ...", end=" ", flush=True)
        meta, df = load_run(run_dir)
        print(f"→ {len(df) if df is not None else 0} epochs  ({meta['run_name']})")
        runs.append((meta, df))
    
    if not runs:
        print("No valid runs found.", file=sys.stderr); sys.exit(1)
    
    print_table(build_param_table(runs), "Run Parameters")
    diff = highlight_diffs(build_param_table(runs))
    if diff:
        print(f"\n  Differing params: {', '.join(sorted(diff))}")
    print_table(build_metric_table(runs), "Final 20% Metrics  (mean ± std)")
    print_table(build_phase_table(runs, args.phase_metric),
                f"Phase Analysis — {args.phase_metric}")
    print(conclusive_summary(runs))
    
    # --- CHANGED: Save individual diagnostic plot per run ---
    if not args.no_plots and any(df is not None for _, df in runs):
        print(f"\n  Generating individual diagnostic charts...")
        for meta, df in runs:
            if df is None:
                continue
            # Save to run_dir itself as eval_diagnostics.png
            output_path = os.path.join(meta["path"], "eval_diagnostics.png")
            print(f"  Saving {meta['run_name']} → {output_path}")
            plot_single_run(meta, df, output_path)
        
        # Optional: Also generate consolidated comparison plot
        if not args.no_comparison and len(runs) > 1:
            # Save comparison plot in parent directory of first run
            first_run_dir = Path(runs[0][0]["path"])
            comparison_path = first_run_dir.parent / "run_comparison.png"
            print(f"\n  Generating comparison chart → {comparison_path} …")
            plot_comparison(runs, str(comparison_path))

if __name__ == "__main__":
    main()
    # Reference legend box
'''
    METRIC INTERPRETATION
    ────────────────────────────
    lifetime    ↑    good
    score_fake  → 0  good
    gap         → 0  good
    disc_reward → 0  good
    value_loss  → 0  good
    sym_loss    → 0  good
    policy_loss: -0.01…-0.05 ✓
    ────────────────────────────
    Imitation errors shown as
    mean ± std (rolling window)
'''