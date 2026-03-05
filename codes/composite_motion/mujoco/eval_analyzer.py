#!/usr/bin/env python3
"""
Training Diagnostics Script — CompositeMotion / ICCGAN (MuJoCo)
================================================================
Accepts run output directories and auto-parses cmds.txt for config metadata.

Usage:
    python training_analyzer.py <run_dir> [<run_dir2> ...]
    python training_analyzer.py checkpoints/jaunty_walk/
    python training_analyzer.py checkpoints/jaunty_walk/ checkpoints/limp_walk/ --window 50

Flags:
    --window   Smoothing window for plots (default 30)
    --out      Output dir for plots. Defaults to first run_dir.
    --terminate-reward  terminate_reward used during training (default -1.0)
    --gamma    PPO gamma (default 0.95)

CSV columns tracked (from helpers.py):
    epoch, lifetime, reward_mean, policy_loss, value_loss, sym_loss
    score_real_{disc_name}, score_fake_{disc_name}   -- one pair per discriminator
    disc_reward_{disc_name}                          -- one per discriminator
    task_reward_{i}                                  -- optional, if goal reward active

    sym_loss in CSV = sym_loss_coeff * raw_symmetry_error  (scaled penalty applied to total loss)
    When sym_loss_coeff=0, sym_loss=0 always (not a training signal, just monitored).

Key env params (from cmds.txt / env_iccgan.py):
    steps_per_cycle  -- control steps per one full motion cycle (max_clip_len * fps)
    episode_length   -- steps_per_cycle x max_cycles
    fps              -- reference motion control frequency
    loop_phase_obs   -- whether phase-conditioned observations are active
    phase_period     -- gait cycle length in seconds

Key training params (from cmds.txt):
    sym_loss_coeff   -- bilateral symmetry loss weight (0 = disabled)
    max_cycles       -- hard reset after this many full motion cycles
    loop_phase_obs   -- phase input flag
"""

import sys, os, re, ast
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ══════════════════════════════════════════════════════════════════════════════
#  METADATA PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _find_balanced_dict(text, start):
    """Return index of the closing brace of the dict/string beginning at `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def parse_cmds_txt(run_dir):
    """
    Parse cmds.txt from run_dir. Returns a metadata dict:
        run_name, training_params, env_params,
        steps_per_cycle, episode_length, max_cycles,
        sym_loss_coeff, loop_phase_obs, fps
    Falls back to sensible defaults when the file is absent or malformed.
    """
    result = {
        "path":            run_dir,
        "run_name":        os.path.basename(run_dir.rstrip("/\\")),
        "training_params": {},
        "env_params":      {},
        "steps_per_cycle": None,
        "episode_length":  None,
        "max_cycles":      5,
        "sym_loss_coeff":  0.0,
        "loop_phase_obs":  False,
        "fps":             30.0,
    }

    cmds_path = os.path.join(run_dir, "cmds.txt")
    if not os.path.exists(cmds_path):
        print(f"  [warn] No cmds.txt found in {run_dir} — using defaults.")
        return result

    with open(cmds_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    # -- Run name: prefer explicit "Run Name:" line (added by updated main.py)
    rn_match = re.search(r"^Run Name:\s*(.+)$", text, re.MULTILINE)
    if rn_match:
        result["run_name"] = rn_match.group(1).strip()
    else:
        # Derive from first command line: find config .py argument
        cmd_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ "(.+?)"', text)
        if cmd_match:
            for token in cmd_match.group(1).split():
                if token.endswith(".py") and "main" not in token.lower():
                    parts = token.replace("\\", "/").split("/")
                    result["run_name"] = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                    break

    # -- Extract LAST occurrence of each param block (last = most recent resume)
    for key, dest in [("Training Params", "training_params"),
                      ("Environment Params", "env_params")]:
        matches = list(re.finditer(re.escape(key) + r": (\{)", text))
        if matches:
            m = matches[-1]
            start = m.start(1)
            end   = _find_balanced_dict(text, start)
            try:
                result[dest] = ast.literal_eval(text[start:end + 1])
            except Exception as e:
                print(f"  [warn] Could not parse {key}: {e}")

    # -- Derive shortcut fields from parsed dicts
    tp = result["training_params"]
    ep = result["env_params"]

    result["steps_per_cycle"] = ep.get("steps_per_cycle")
    result["episode_length"]  = ep.get("episode_length", tp.get("episode_length"))
    result["max_cycles"]      = int(tp.get("max_cycles", ep.get("max_cycles", 5)))
    result["sym_loss_coeff"]  = float(tp.get("sym_loss_coeff", 0.0))
    result["loop_phase_obs"]  = bool(tp.get("loop_phase_obs", False))
    result["fps"]             = float(ep.get("fps", 30.0))

    # Fallback: compute episode_length if only steps_per_cycle is known
    if result["episode_length"] is None and result["steps_per_cycle"]:
        result["episode_length"] = result["steps_per_cycle"] * result["max_cycles"]

    return result


def load_run(run_dir):
    """Return (df, metadata) for a run directory."""
    meta     = parse_cmds_txt(run_dir)
    csv_path = os.path.join(run_dir, "training_metrics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No training_metrics.csv in {run_dir}")
    df = pd.read_csv(csv_path)
    df["_label"] = meta["run_name"]
    print(f"  Loaded '{meta['run_name']}':  {len(df)} entries, "
          f"epochs {df.epoch.min():.0f}–{df.epoch.max():.0f}  │  "
          f"steps_per_cycle={meta['steps_per_cycle']}  "
          f"episode_length={meta['episode_length']}  "
          f"sym_coeff={meta['sym_loss_coeff']}  "
          f"phase_obs={meta['loop_phase_obs']}")
    return df, meta


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="ICCGAN per-run training diagnostics")
parser.add_argument("run_dirs", nargs="+",
                    help="Run output directory(ies) containing training_metrics.csv + cmds.txt")
parser.add_argument("--window",            type=int,   default=30,   help="Smoothing window (default 30)")
parser.add_argument("--terminate-reward",  type=float, default=-1.0, help="terminate_reward used in training")
parser.add_argument("--gamma",             type=float, default=0.95, help="PPO gamma (default 0.95)")
parser.add_argument("--out",               default=None, help="Output directory for plots")
args = parser.parse_args()

out_dir = args.out or os.path.dirname(os.path.abspath(args.run_dirs[0])) # parent dir
os.makedirs(out_dir, exist_ok=True)

TERMINATE_REWARD = args.terminate_reward
GAMMA            = args.gamma
W                = args.window

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD RUNS
# ══════════════════════════════════════════════════════════════════════════════

datasets = []
metas    = []
print("\nLoading runs …")
for d in args.run_dirs:
    df, meta = load_run(d)
    datasets.append(df)
    metas.append(meta)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def smooth(s, w):
    return s.rolling(w, min_periods=1, center=True).mean()

def slope_of(s):
    if len(s) < 3:
        return float("nan")
    return float(np.polyfit(np.arange(len(s)), s.values, 1)[0])

def phase_slopes(df, col):
    n, t = len(df), max(1, len(df) // 3)
    def sl(sub):
        if len(sub) < 3:
            return float("nan")
        return float(np.polyfit(np.arange(len(sub)), sub[col].values, 1)[0])
    return sl(df.iloc[:t]), sl(df.iloc[t:2*t]), sl(df.iloc[2*t:])

def quartile_means(s):
    n, q = len(s), max(1, len(s) // 4)
    return [s.iloc[i*q:min((i+1)*q, n)].mean() for i in range(4)]

def sign_label(v, pos_good=True):
    if np.isnan(v):
        return "n/a"
    tag = f"{v:+.5f}/ep"
    if pos_good:
        return tag + (" ↑ good" if v > 1e-6 else (" ↓ bad" if v < -1e-6 else " → flat"))
    else:
        return tag + (" ↓ good" if v < -1e-6 else (" ↑ bad" if v > 1e-6 else " → flat"))

def accel(s):
    if len(s) < 10:
        return float("nan")
    t = len(s) // 2
    return slope_of(s.iloc[t:]) - slope_of(s.iloc[:t])

def mean_std_str(s, fmt=".4f"):
    """Format a series as 'mean ± std'."""
    return f"{s.mean():{fmt}} ± {s.std():{fmt}}"

def tail_pct(df, frac=0.2):
    return df.iloc[max(0, int(len(df) * (1 - frac))):]

def sep(): print("=" * 78)
def sub(): print("-" * 78)


# ══════════════════════════════════════════════════════════════════════════════
#  PER-RUN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

all_reports = {}

for df, meta in zip(datasets, metas):
    label      = meta["run_name"]
    spc        = meta["steps_per_cycle"]
    ep_len     = meta["episode_length"]
    max_cycles = meta["max_cycles"]
    sym_coeff  = meta["sym_loss_coeff"]
    loop_phase = meta["loop_phase_obs"]

    cycle_t = spc if spc else (ep_len or 300)
    ep_t    = ep_len or (cycle_t * max_cycles)

    rep = {}
    all_reports[label] = rep

    # Dynamic column detection
    disc_r_cols  = [c for c in df.columns if c.startswith("disc_reward_")]
    score_r_cols = [c for c in df.columns if c.startswith("score_real_")]
    score_f_cols = [c for c in df.columns if c.startswith("score_fake_")]
    task_r_cols  = [c for c in df.columns if c.startswith("task_reward_")]
    has_sym = "sym_loss" in df.columns and df["sym_loss"].notna().any()

    sep()
    print(f"\n  RUN: {label}   (epochs {df.epoch.min():.0f}–{df.epoch.max():.0f})")
    sep()

    # ── [0] RUN METADATA ─────────────────────────────────────────────────────
    print(f"\n[0] RUN METADATA")
    print(f"    Config file     : {label}")
    print(f"    steps_per_cycle : {spc or '(unknown)'}  "
          f"│  episode_length : {ep_t}  "
          f"│  max_cycles : {max_cycles}")
    ep = meta["env_params"]
    print(f"    fps             : {meta['fps']:.0f}  "
          f"│  phase_period : {ep.get('phase_period','?')}s  "
          f"│  loop_phase_obs : {loop_phase}")
    print(f"    sym_loss_coeff  : {sym_coeff}  "
          f"({'ACTIVE — shapes total loss' if sym_coeff > 0 else 'disabled (0) — monitored only'})")
    tp = meta["training_params"]
    if tp:
        key_tp = ["actor_lr","critic_lr","disc_lr","horizon","num_envs",
                  "batch_size","gamma","terminate_reward"]
        print("    Training params : " +
              "  ".join(f"{k}={tp[k]}" for k in key_tp if k in tp))

    # ── [1] SURVIVAL ─────────────────────────────────────────────────────────
    lt = df["lifetime"]
    rep.update({
        "lt_mean":    lt.mean(),
        "lt_max":     lt.max(),
        "lt_final10": lt.tail(10).mean(),
        "lt_slope":   slope_of(lt),
        "lt_accel":   accel(lt),
        "lt_q":       quartile_means(lt),
    })
    es_l, em_l, el_l = phase_slopes(df, "lifetime")
    pct_cycle = rep["lt_final10"] / cycle_t * 100
    pct_ep    = rep["lt_final10"] / ep_t    * 100

    print(f"\n[1] SURVIVAL — steps survived before fall / hard reset per env")
    print(f"    Meaning: higher = agent stays balanced longer.")
    print(f"    Target: survive one full cycle ({cycle_t} steps), "
          f"then full episode ({ep_t} steps = {max_cycles}× cycles).")
    print(f"    mean ± std : {mean_std_str(lt)}  "
          f"│  max={lt.max():.1f}  │  final-10-avg={rep['lt_final10']:.1f}")
    print(f"    Coverage   : {pct_cycle:.0f}% of one cycle  /  {pct_ep:.0f}% of episode")
    print(f"    Quartiles  : {[f'{v:.1f}' for v in rep['lt_q']]}")
    print(f"    Overall slope  : {sign_label(rep['lt_slope'], pos_good=True)}")
    print(f"    Phase slopes   : early={es_l:+.4f}  mid={em_l:+.4f}  late={el_l:+.4f}  /ep")
    print(f"    Acceleration   : {rep['lt_accel']:+.4f}  "
          f"({'faster in 2nd half ✓' if rep['lt_accel'] > 0 else 'decelerating ⚠'})")

    if pct_ep < 10:
        diag = ("⚠️  CRITICAL – falls within <10% of episode. "
                "Physics or actuator issue. Check kp, contact thresholds.")
    elif pct_cycle < 50:
        diag = ("⚠️  POOR – surviving <50% of one cycle. "
                "Discriminator dominating; increase exploration or tune reward scale.")
    elif pct_cycle < 100:
        diag = "⚡ PARTIAL – ~0.5–1 motion cycle. Imitation starting; expect gradual improvement."
    elif pct_cycle < 150:
        diag = "⚡ APPROACHING CYCLE – close to one full cycle. Policy learning the motion."
    elif pct_ep < 80:
        diag = "✓  DECENT – partial episode survival, multi-cycle looping occurring."
    else:
        diag = "✓✓ GOOD – agent survives most / all of the episode."
    print(f"    → {diag}")

    # ── [2] DISCRIMINATOR ────────────────────────────────────────────────────
    print(f"\n[2] DISCRIMINATOR — real vs fake GAN scores")
    print(f"    score_real: disc confidence on reference motion  (want: stable 0.15–0.25)")
    print(f"    score_fake: disc confidence on policy rollout    (want: rising toward score_real)")
    print(f"    gap = real − fake.  Want → 0.  Growing = disc winning.  Shrinking = policy winning.")

    for sr_col, sf_col in zip(score_r_cols, score_f_cols):
        name = sr_col.replace("score_real_", "")
        sr, sf = df[sr_col], df[sf_col]
        gap    = sr - sf
        t20    = tail_pct(df, 0.2)

        rep[f"sr_{name}_f10"]  = sr.tail(10).mean()
        rep[f"sf_{name}_f10"]  = sf.tail(10).mean()
        rep[f"gap_{name}_f10"] = gap.tail(10).mean()

        es_r, em_r, el_r = phase_slopes(df, sr_col)
        es_f, em_f, el_f = phase_slopes(df, sf_col)
        acc_r, acc_f     = accel(sr), accel(sf)

        print(f"\n    Discriminator [{name}]")
        print(f"      score_real  (final-20%): {mean_std_str(sr.iloc[len(sr)-len(t20):])}")
        print(f"        Trend: {sign_label(slope_of(sr), pos_good=False)}")
        print(f"        Phase: early={es_r:+.5f}  mid={em_r:+.5f}  late={el_r:+.5f}  accel={acc_r:+.4f}")
        print(f"        +ve slope = disc better at recognising real data. "
              f"Should stabilise; continuously rising = disc diverging.")
        print(f"      score_fake  (final-20%): {mean_std_str(sf.iloc[len(sf)-len(t20):])}")
        print(f"        Trend: {sign_label(slope_of(sf), pos_good=True)}")
        print(f"        Phase: early={es_f:+.5f}  mid={em_f:+.5f}  late={el_f:+.5f}  accel={acc_f:+.4f}")
        print(f"        +ve slope = policy fooling disc more = GOOD.")
        print(f"      gap (final-20%): {mean_std_str(gap.iloc[len(gap)-len(t20):])}")
        print(f"        Trend: {sign_label(slope_of(gap), pos_good=False)}")
        print(f"        gap < 0.05 = near-perfect imitation  │  gap > 0.5 = disc dominating")

        gsl, fsl = slope_of(gap), slope_of(sf)
        if gsl > 1e-4 and fsl < 5e-5:
            print(f"      → ⚠️  DISC WINNING: gap growing AND fake score flat.")
        elif gsl > 1e-4:
            print(f"      → ⚡ DISC AHEAD but fake score rising — policy IS learning, slowly.")
        else:
            print(f"      → ✓  Gap stable / shrinking — policy is catching up.")

    # ── [3] REWARD ───────────────────────────────────────────────────────────
    rw = df["reward_mean"]
    print(f"\n[3] REWARD DECOMPOSITION")
    print(f"    Meaning: disc_reward = log(score_fake / (1−score_fake)).")
    print(f"             0 = perfectly fools disc.  −ve = disc confident rollout is fake.")
    print(f"    reward_mean: {mean_std_str(rw)}  │  "
          f"slope={sign_label(slope_of(rw), pos_good=True)}")
    fp = df[rw > 0]
    if not fp.empty:
        print(f"    First +ve reward at epoch {int(fp.iloc[0].epoch)}  "
              f"(value={fp.iloc[0].reward_mean:.4f})")
    es_rw, em_rw, el_rw = phase_slopes(df, "reward_mean")
    print(f"    Phase: early={es_rw:+.5f}  mid={em_rw:+.5f}  late={el_rw:+.5f}  /ep")
    for col in disc_r_cols:
        name  = col.replace("disc_reward_", "")
        t20s  = df[col].iloc[max(0, int(len(df)*0.8)):]
        rep[f"dr_{name}_f10"]   = df[col].tail(10).mean()
        rep[f"dr_{name}_slope"] = slope_of(df[col])
        print(f"    disc_reward [{name}] (final-20%): {mean_std_str(t20s)}  "
              f"slope={slope_of(df[col]):+.6f}/ep  accel={accel(df[col]):+.4f}")
    if task_r_cols:
        print(f"    Task rewards (final-10):")
        for col in task_r_cols:
            print(f"      {col}: {mean_std_str(df[col].tail(10))}")

    # ── [4] VALUE FUNCTION ───────────────────────────────────────────────────
    vl = df["value_loss"]
    rep["vl_f10"]   = vl.tail(10).mean()
    rep["vl_slope"] = slope_of(vl)
    t20_vl = tail_pct(vl, 0.2)
    print(f"\n[4] VALUE FUNCTION — critic MSE loss  (want ↓ → 0)")
    print(f"    Meaning: falling = critic learning actual reward landscape.")
    print(f"    final-20%: {mean_std_str(t20_vl)}  │  "
          f"slope={sign_label(rep['vl_slope'], pos_good=False)}")
    es_v, em_v, el_v = phase_slopes(df, "value_loss")
    print(f"    Phase: early={es_v:+.5f}  mid={em_v:+.5f}  late={el_v:+.5f}  /ep")
    if rep["vl_f10"] < 0.005:
        print(f"    → ✓  Near-zero: critic converged. Verify reward is meaningful.")
    elif rep["vl_f10"] < 0.05:
        print(f"    → ✓  Low and decreasing: critic converging well.")
    elif rep["vl_f10"] > 1.0:
        print(f"    → ⚠️  HIGH (>1.0): critic unstable. Reduce critic_lr or check reward spikes.")
    else:
        print(f"    → ⚡ Moderate. Still converging — normal mid-training.")

    # ── [5] POLICY (PPO) ─────────────────────────────────────────────────────
    pl = df["policy_loss"]
    rep["pl_f10"]   = pl.tail(10).mean()
    rep["pl_slope"] = slope_of(pl)
    print(f"\n[5] POLICY LOSS (PPO clipped)  (want moderate −ve)")
    print(f"    Meaning: −0.01 to −0.05 = healthy updates.  >0 or <−0.1 = issues.")
    print(f"    early (first-20%): {mean_std_str(tail_pct(pl.iloc[:int(len(pl)*0.2)], 1.0))}")
    print(f"    final-20%        : {mean_std_str(tail_pct(pl, 0.2))}  │  "
          f"slope={sign_label(rep['pl_slope'], pos_good=False)}")
    es_p, em_p, el_p = phase_slopes(df, "policy_loss")
    print(f"    Phase: early={es_p:+.5f}  mid={em_p:+.5f}  late={el_p:+.5f}  /ep")
    if abs(rep["pl_f10"]) < 0.001:
        print(f"    → ⚠️  Near-zero: policy stalled. Consider entropy bonus or LR schedule.")
    elif rep["pl_f10"] < -0.1:
        print(f"    → ⚠️  Very negative (<−0.1): large updates. Watch for mode collapse.")
    else:
        print(f"    → ✓  Moderately negative: healthy PPO behaviour.")

    # ── [6] SYMMETRY LOSS ────────────────────────────────────────────────────
    print(f"\n[6] BILATERAL SYMMETRY LOSS")
    if has_sym:
        sl_col = df["sym_loss"]
        if sym_coeff > 0:
            raw      = sl_col / sym_coeff      # actual pose asymmetry, coefficient-free
            t20_raw  = tail_pct(raw, 0.2)
            raw_slope = slope_of(raw)
            print(f"    sym_loss_coeff = {sym_coeff}  → ACTIVE (contributes to total loss)")
            print(f"    CSV stores: coeff × raw_error  =  sym_loss_coeff × [(mu_R − sign·mu_L)²].mean()")
            print(f"    Scaled loss (from CSV) final-20%: {mean_std_str(tail_pct(sl_col, 0.2))}")
            print(f"    Raw pose asymmetry (÷{sym_coeff}) final-20%: {mean_std_str(t20_raw)}")
            print(f"    Raw slope: {sign_label(raw_slope, pos_good=False)}")
            es_s, em_s, el_s = phase_slopes(df, "sym_loss")
            print(f"    Phase (scaled): early={es_s:+.6f}  mid={em_s:+.6f}  late={el_s:+.6f}  /ep")
            if raw_slope < -1e-8:
                print(f"    → ✓  Decreasing: policy learning bilateral symmetry.")
            elif abs(raw_slope) < 1e-8:
                print(f"    → ⚡ Flat: symmetry stable but not actively improving.")
            else:
                print(f"    → ⚠️  Increasing: policy deviating from symmetry.")
            rep["sym_raw_f10"] = raw.tail(10).mean()
        else:
            print(f"    sym_loss_coeff = 0.0  → DISABLED (monitoring only — not applied to loss)")
            print(f"    Raw pose asymmetry (full): {mean_std_str(sl_col)}")
            print(f"    Trend: {sign_label(slope_of(sl_col), pos_good=False)}  (informational only)")
            rep["sym_raw_f10"] = sl_col.tail(10).mean()
    else:
        print(f"    sym_loss column absent in CSV — not applicable for this run.")

    # ── [7] TERMINATE-REWARD SAFETY CHECK ───────────────────────────────────
    print(f"\n[7] TERMINATE-REWARD SAFETY CHECK")
    print(f"    If adv(fall) = terminate_reward − V_long_horizon > 0 → agent prefers falling.")
    if disc_r_cols:
        dr        = df[disc_r_cols[0]]
        V_long    = dr / (1 - GAMMA)
        adv_fall  = TERMINATE_REWARD - V_long
        bad       = df[adv_fall > 0]
        if not bad.empty:
            r0 = bad.iloc[0]
            print(f"    ⚠️  Falling first incentivised at epoch {int(r0.epoch)}  "
                  f"(disc_r={r0[disc_r_cols[0]]:.4f}, adv(fall)={adv_fall.loc[r0.name]:+.3f})")
        else:
            print(f"    ✓  terminate_reward={TERMINATE_REWARD} keeps falling penalised throughout.")
        final_adv = adv_fall.iloc[-1]
        safe_tr   = V_long.min() - 2.0
        print(f"    adv(fall) at final epoch: {final_adv:+.3f}  "
              f"({'⚠ incentivised' if final_adv > 0 else '✓ penalised'})")
        if safe_tr < TERMINATE_REWARD:
            print(f"    → Recommended terminate_reward ≤ {max(-20.0, safe_tr-1):.0f}  "
                  f"(worst case: {dr.min():.3f} / {1-GAMMA:.2f} = {dr.min()/(1-GAMMA):.1f})")

    # ── [8] PLATEAU / COLLAPSE DETECTION ────────────────────────────────────
    print(f"\n[8] PLATEAU / COLLAPSE DETECTION  (smoothing window = {W})")
    lt_roll_mean = lt.rolling(W, min_periods=1).mean()
    lt_roll_std  = lt.rolling(W, min_periods=1).std().fillna(0)
    collapse     = (lt_roll_mean.diff() < -0.5) & (lt_roll_mean < 15)
    n_coll       = int(collapse.sum())
    last30       = df.iloc[int(len(df) * 0.7):]["lifetime"]
    lt_range_l30 = last30.max() - last30.min()
    rep["n_collapse"]   = n_coll
    rep["lt_range_l30"] = lt_range_l30
    print(f"    Rolling std(lifetime): {lt_roll_std.mean():.2f} mean  (higher = noisier training)")
    print(f"    Collapse events (drop>0.5 while mean_lt<15): {n_coll}")
    if lt_range_l30 < 1.0:
        print(f"    → ⚠️  STAGNATION: variation in last 30% = {lt_range_l30:.2f} (no progress).")
    else:
        print(f"    → Lifetime range in last 30%: {lt_range_l30:.2f}  "
              f"({'still active' if lt_range_l30 > 3 else 'stabilising'}).")

    # ── Notes on new methodology flags ──────────────────────────────────────
    if loop_phase or sym_coeff > 0:
        print(f"\n[NOTE] Active methodology flags for this run:")
        if loop_phase:
            pp = meta["env_params"].get("phase_period", "?")
            print(f"       loop_phase_obs=True  │  phase_period={pp}s  ({spc or '?'} steps/cycle)")
            print(f"       → Phase-conditioned observations provide the policy with temporal "
                  f"position within the gait cycle. Expect faster cycle learning vs baseline.")
        if sym_coeff > 0:
            print(f"       sym_loss_coeff={sym_coeff} — bilateral symmetry regularization is active.")
            print(f"       → Penalises asymmetric mean actions across L/R joint pairs. "
                  f"Helps generalise to mirrored motions and stabilises gait.")


# ══════════════════════════════════════════════════════════════════════════════
#  NARRATIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

sep()
print("\n  NARRATIVE SUMMARY  (all runs)")
sep()

for meta, rep in zip(metas, all_reports.values()):
    label      = meta["run_name"]
    spc        = meta["steps_per_cycle"]
    ep_len     = meta["episode_length"]
    cycle_t    = spc if spc else (ep_len or 300)
    ep_t       = ep_len or cycle_t * meta["max_cycles"]
    df         = datasets[metas.index(meta)]

    pct_cycle  = rep["lt_final10"] / cycle_t * 100
    pct_ep     = rep["lt_final10"] / ep_t    * 100

    dr_f10_keys   = [k for k in rep if k.startswith("dr_") and k.endswith("_f10")]
    dr_slope_keys = [k for k in rep if k.startswith("dr_") and k.endswith("_slope")]
    disc_r_latest = rep[dr_f10_keys[0]] if dr_f10_keys else float("nan")
    disc_r_slope  = rep[dr_slope_keys[0]] if dr_slope_keys else float("nan")

    sub()
    print(f"\n[{label}]")
    print(f"  Epochs       : {df.epoch.min():.0f} – {df.epoch.max():.0f}")
    print(f"  Lifetime     : final-10 avg = {rep['lt_final10']:.1f} steps  "
          f"({pct_cycle:.0f}% of cycle / {pct_ep:.0f}% of episode)  "
          f"slope={rep['lt_slope']:+.4f}/ep")

    dr_col = next((c for c in df.columns if c.startswith("disc_reward_")), None)
    if dr_col:
        t20 = tail_pct(df[dr_col], 0.2)
        print(f"  Disc reward  : {mean_std_str(t20)} (final-20%)  "
              f"slope={disc_r_slope:+.6f}/ep  "
              f"({'rising ✓' if disc_r_slope > 0 else 'falling ⚠'})")
    print(f"  Value loss   : {rep['vl_f10']:.5f}  slope={rep['vl_slope']:+.6f}/ep")
    if "sym_raw_f10" in rep:
        print(f"  Sym (raw)    : {rep['sym_raw_f10']:.6f}  (coeff={meta['sym_loss_coeff']})")
    print(f"  Collapses    : {rep['n_collapse']}")

    positives, issues = [], []

    if pct_ep < 10:
        issues.append("CRITICAL: <10% episode survived — physics or reward fundamentally broken.")
    elif pct_cycle < 50:
        issues.append("Lifetime <50% of one cycle — policy barely holding posture.")
    else:
        positives.append("Lifetime adequate — posture partially maintained.")

    gap_keys = [k for k in rep if k.startswith("gap_") and k.endswith("_f10")]
    sf_keys  = [k for k in rep if k.startswith("sf_")  and k.endswith("_f10")]
    if gap_keys and sf_keys:
        gf10, sf10 = rep[gap_keys[0]], rep[sf_keys[0]]
        if sf10 > -0.01:
            positives.append(f"score_fake={sf10:.3f} ≈ 0 → nearly fooling discriminator.")
        elif sf10 > -0.05:
            positives.append(f"score_fake={sf10:.3f} trending up — slow but real progress.")
        else:
            issues.append(f"score_fake={sf10:.3f} — policy still far from reference motion.")
        if gf10 < 0.1:
            positives.append(f"Gap={gf10:.3f} — near-perfect imitation.")
        elif gf10 < 0.35:
            positives.append(f"Gap={gf10:.3f} — stabilising; policy competing with disc.")
        else:
            issues.append(f"Gap={gf10:.3f} — discriminator dominating.")

    if not np.isnan(disc_r_slope):
        if disc_r_slope > 0:
            positives.append(f"disc_reward rising → imitation quality improving over time.")
        else:
            issues.append("disc_reward trending down — policy losing ground to discriminator.")

    if rep["vl_f10"] < 0.05:
        positives.append(f"Value loss={rep['vl_f10']:.5f} — critic stable.")
    else:
        issues.append(f"Value loss={rep['vl_f10']:.5f} — critic still converging.")

    if meta["loop_phase_obs"]:
        positives.append("loop_phase_obs=True — phase signal assists cycle learning.")
    if meta["sym_loss_coeff"] > 0:
        positives.append(f"sym_loss_coeff={meta['sym_loss_coeff']} — symmetry regularization active.")

    print()
    for p in positives: print(f"  ✓ {p.strip()}")
    for i in issues:    print(f"  ⚠ {i.strip()}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

colors  = plt.cm.tab10.colors
has_sym = any("sym_loss" in df.columns and df["sym_loss"].notna().any() for df in datasets)
n_rows  = 5 if has_sym else 4

fig = plt.figure(figsize=(23, 4 * n_rows + 2))
fig.suptitle("ICCGAN Training Diagnostics", fontsize=14, fontweight="bold", y=0.999)

gs = gridspec.GridSpec(n_rows, 3, hspace=0.55, wspace=0.38,
                       left=0.06, right=0.97, top=0.97, bottom=0.03)

ax_lt   = fig.add_subplot(gs[0, :2])
ax_lt_z = fig.add_subplot(gs[0, 2])
ax_disc = fig.add_subplot(gs[1, :2])
ax_fake = fig.add_subplot(gs[1, 2])
ax_dr   = fig.add_subplot(gs[2, :2])
ax_dr_z = fig.add_subplot(gs[2, 2])
ax_vl   = fig.add_subplot(gs[3, 0])
ax_pl   = fig.add_subplot(gs[3, 1])
ax_gap  = fig.add_subplot(gs[3, 2])
if has_sym:
    ax_sym = fig.add_subplot(gs[4, :])


def fmt_ax(ax, title, ylabel="", hline=None, note=None, legend=True):
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
    ax.set_xlabel("Epoch", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.18)
    if legend:
        ax.legend(fontsize=6, loc="best")
    if hline is not None:
        ax.axhline(hline, color="gray", linestyle=":", lw=0.8)
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes,
                fontsize=5.5, va="top", color="#444444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))


for i, (df, meta) in enumerate(zip(datasets, metas)):
    c     = colors[i % len(colors)]
    ep    = df["epoch"]
    n     = len(df)
    spc   = meta["steps_per_cycle"]
    label = meta["run_name"]
    cut   = int(n * 0.7)

    dr_col = next((x for x in df.columns if x.startswith("disc_reward_")), None)
    sr_col = next((x for x in df.columns if x.startswith("score_real_")),  None)
    sf_col = next((x for x in df.columns if x.startswith("score_fake_")),  None)

    # Lifetime (full + zoomed)
    lt_s = smooth(df.lifetime, W)
    ax_lt.plot(ep, lt_s, label=label, color=c, lw=1.5)
    ax_lt.fill_between(ep,
        df.lifetime.rolling(W, min_periods=1).quantile(0.1),
        df.lifetime.rolling(W, min_periods=1).quantile(0.9),
        alpha=0.12, color=c)
    if spc:
        ax_lt.axhline(spc, color=c, linestyle=":", lw=1, alpha=0.55,
                      label=f"{label} cycle={spc}")
    ax_lt_z.plot(ep.iloc[cut:], lt_s.iloc[cut:], color=c, lw=1.5, label=label)
    ax_lt_z.fill_between(ep.iloc[cut:],
        df.lifetime.iloc[cut:].rolling(W, min_periods=1).quantile(0.1),
        df.lifetime.iloc[cut:].rolling(W, min_periods=1).quantile(0.9),
        alpha=0.15, color=c)
    if spc:
        ax_lt_z.axhline(spc, color=c, linestyle=":", lw=0.8)

    # Discriminator scores
    if sr_col and sf_col:
        ax_disc.plot(ep, smooth(df[sr_col], W), label=f"{label} real",
                     color=c, lw=1.5)
        ax_disc.plot(ep, smooth(df[sf_col], W), label=f"{label} fake",
                     color=c, lw=1.5, ls="--")
        gap = df[sr_col] - df[sf_col]
        ax_gap.plot(ep, smooth(gap, W), color=c, label=label, lw=1.5)
        ax_fake.plot(ep, smooth(df[sf_col], W), color=c, label=label, lw=1.5)

    # Disc reward
    if dr_col:
        ax_dr.plot(ep, smooth(df[dr_col], W), color=c, label=label, lw=1.5)
        ax_dr_z.plot(ep.iloc[cut:], smooth(df[dr_col], W).iloc[cut:],
                     color=c, label=label, lw=1.5)
        ax_dr_z.fill_between(ep.iloc[cut:],
            df[dr_col].iloc[cut:].rolling(W, min_periods=1).quantile(0.1),
            df[dr_col].iloc[cut:].rolling(W, min_periods=1).quantile(0.9),
            alpha=0.12, color=c)

    # Value loss (log scale)
    vl_cl = df.value_loss.clip(lower=1e-8)
    ax_vl.semilogy(ep, vl_cl, alpha=0.12, color=c)
    ax_vl.semilogy(ep, smooth(vl_cl, W), color=c, label=label, lw=1.5)

    # Policy loss
    ax_pl.plot(ep, smooth(df.policy_loss, W), color=c, label=label, lw=1.5)

    # Symmetry loss (show raw = CSV / coeff when coeff > 0)
    if has_sym and "sym_loss" in df.columns:
        sym_coeff = meta["sym_loss_coeff"]
        sym_vals  = df["sym_loss"].copy()
        if sym_coeff > 0:
            sym_vals = sym_vals / sym_coeff
        sym_label = f"{label} (raw)" if sym_coeff > 0 else label
        ax_sym.plot(ep, smooth(sym_vals, W), color=c, label=sym_label, lw=1.5)
        ax_sym.fill_between(ep,
            sym_vals.rolling(W, min_periods=1).quantile(0.1),
            sym_vals.rolling(W, min_periods=1).quantile(0.9),
            alpha=0.1, color=c)


# Format all axes
fmt_ax(ax_lt,   "Lifetime  (dotted lines = steps_per_cycle target per run)", "Steps")
fmt_ax(ax_lt_z, "Lifetime  (last 30% zoomed)", "Steps")
fmt_ax(ax_disc, "Disc Scores: Real (solid) vs Fake (dashed)", "Score", hline=0,
       note="Real stable ~0.15–0.25 │ Fake rising toward Real = learning")
fmt_ax(ax_fake, "Fake Score only\n(rising toward 0 = policy fooling disc)", "Score", hline=0)
fmt_ax(ax_dr,   "Disc Reward  (want → 0, rising = policy improving)", "Reward", hline=0,
       note="−ve = disc winning  │  0 = perfect imitation")
fmt_ax(ax_dr_z, "Disc Reward  (last 30% zoomed)", "Reward", hline=0)
fmt_ax(ax_vl,   "Value Loss  (log scale, want ↓ → 0)", "Loss")
fmt_ax(ax_pl,   "Policy Loss (PPO)\n−0.01 to −0.05 = healthy", "Loss", hline=0)
fmt_ax(ax_gap,  "Score Gap = real−fake  (want → 0)", "Gap", hline=0,
       note="Shrinking = policy winning  │  > 0.5 = disc dominating")
if has_sym:
    fmt_ax(ax_sym,
           "Symmetry Loss  (raw pose asymmetry = CSV ÷ coeff when coeff>0)  |  want ↓",
           "Raw Asymmetry")

# Reference legend box
ref_text = (
    "METRIC INTERPRETATION\n"
    "────────────────────────────\n"
    "lifetime    ↑  good\n"
    "score_fake  → 0  good\n"
    "gap         → 0  good\n"
    "disc_reward → 0  good\n"
    "value_loss  → 0  good\n"
    "sym_loss    → 0  good\n"
    "policy_loss: −0.01…−0.05 ✓\n"
    "────────────────────────────\n"
    "Imitation errors shown as\n"
    "mean ± std (rolling window)"
)
ax_ann = fig.add_axes([0.005, 0.002, 0.20, 0.10])
ax_ann.axis("off")
ax_ann.text(0.0, 0.5, ref_text, fontsize=5.5, va="center", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

out_path = os.path.join(out_dir, "eval_diagnostics.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\n→ Diagnostics plot saved: {out_path}")

sep()
print("DONE  — eval_analyzer.py")
sep()