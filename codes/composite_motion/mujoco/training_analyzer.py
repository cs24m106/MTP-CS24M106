"""
Training Diagnostics Script — CompositeMotion / ICCGAN (MuJoCo)
================================================================
Usage:
  python analyze_training.py <path/to/training_metrics.csv> [<path2.csv> ...]
  python analyze_training.py checkpoints/kick/training_metrics.csv
  python analyze_training.py kick.csv walk.csv --labels kick walk
  python analyze_training.py checkpoints/jaunty_walk/training_metrics.csv \\
      --motion assets/motions/iccgan/jaunty_walk.json

Flags:
  --motion   Path to the .json motion file. Used to compute correct motion-clip
             lifetime target (clip_steps = num_frames, since control Hz = motion Hz).
  --episode-length  Episode length in steps (default 300). Ignored if --motion given.
  --terminate-reward  Actual terminate_reward used during training (default -1).
  --gamma    PPO gamma used in training (default 0.95).
  --labels   Legend names per CSV file.
  --window   Smoothing window for plots (default 30).
  --out      Output directory for plots. Defaults to same folder as first CSV.

What this script diagnoses:
  1.  Survival       – lifetime vs motion-clip length and episode length
  2.  Discriminator  – real/fake score gap, convergence, policy catching up
  3.  Reward         – disc_reward trend; when did reward turn +ve?
  4.  Value function – critic convergence quality
  5.  Policy (PPO)   – policy_loss interpretation
  6.  Termination    – is falling incentivised over surviving?
  7.  Training phases – early / mid / late slope per metric
  8.  Plateau/collapse detection
  9.  Full narrative summary with next-action recommendations
"""

import sys, os, json, glob, re
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ICCGAN training diagnostics")
parser.add_argument("csvfiles", nargs="+", help="One or more training_metrics.csv files")
parser.add_argument("--labels", nargs="*", default=None)
parser.add_argument("--motion", default=None,
                    help="Path to motion JSON file to determine clip length target")
parser.add_argument("--episode-length", type=int, default=300,
                    help="Episode length in control steps (default 300)")
parser.add_argument("--terminate-reward", type=float, default=-1.0,
                    help="terminate_reward used during training (default -1.0)")
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--window", type=int, default=30)
parser.add_argument("--out", default=None,
                    help="Output dir for plots. Defaults to folder of first CSV.")
args = parser.parse_args()

# ── Output dir: default = same folder as first CSV ────────────────────────────
if args.out is None:
    args.out = os.path.dirname(os.path.abspath(args.csvfiles[0]))
os.makedirs(args.out, exist_ok=True)

labels = args.labels or [os.path.basename(os.path.dirname(f)) or f for f in args.csvfiles]
if len(labels) < len(args.csvfiles):
    labels += [f"run_{i}" for i in range(len(labels), len(args.csvfiles))]

# ── Motion-aware lifetime target ──────────────────────────────────────────────
EPISODE_LEN  = args.episode_length
MOTION_FILE  = args.motion
CLIP_STEPS   = None   # steps per one motion cycle at 30Hz control

# Try to auto-detect motion file from CSV path if not given
if MOTION_FILE is None:
    csv_dir = os.path.dirname(os.path.abspath(args.csvfiles[0]))
    run_name = os.path.basename(csv_dir)
    # Look for motion JSON in common relative locations
    candidates = (
        glob.glob(os.path.join(csv_dir, "../../assets/motions/**", f"{run_name}.json"), recursive=True) +
        glob.glob(os.path.join(csv_dir, "../../assets/motions/iccgan", f"{run_name}.json"))
    )
    if candidates:
        MOTION_FILE = candidates[0]
        print(f"[auto-detected motion file]: {MOTION_FILE}")

if MOTION_FILE and os.path.isfile(MOTION_FILE):
    try:
        with open(MOTION_FILE) as fh:
            mdata = json.load(fh)
        fps = float(mdata.get("fps", 30))
        # Try different key names for frame count
        num_frames = (mdata.get("num_frames") or
                      len(mdata.get("frames", [])) or
                      len(mdata.get("Frames", [])))
        if num_frames > 0:
            clip_seconds = num_frames / fps
            control_hz   = 30  # assumed; matches sim_speed/frameskip=120/4
            CLIP_STEPS   = int(num_frames)
            print(f"[motion]: {os.path.basename(MOTION_FILE)} — "
                  f"{num_frames} frames @ {fps:.0f} fps = {clip_seconds:.2f}s = {CLIP_STEPS} control steps/cycle")
    except Exception as e:
        print(f"[warn] Could not parse motion file: {e}")

if CLIP_STEPS is None:
    print(f"[note] No motion file found — using episode_length={EPISODE_LEN} as lifetime target.")

# ── Load CSVs ─────────────────────────────────────────────────────────────────
datasets = []
for path, label in zip(args.csvfiles, labels):
    df = pd.read_csv(path)
    df["_label"] = label
    datasets.append(df)
    print(f"Loaded '{label}': {len(df)} entries, epochs {df.epoch.min():.0f}–{df.epoch.max():.0f}")

TERMINATE_REWARD = args.terminate_reward
GAMMA            = args.gamma

# ── Helper functions ──────────────────────────────────────────────────────────
def smooth(s, w):
    return s.rolling(w, min_periods=1, center=True).mean()

def slope_of(s):
    """Overall linear slope per epoch."""
    if len(s) < 3: return float("nan")
    x = np.arange(len(s))
    return float(np.polyfit(x, s.values, 1)[0])

def phase_slopes(df, col):
    """Return slope in early (0-33%), mid (33-66%), late (66-100%) phases."""
    n = len(df)
    t = n // 3
    def sl(sub):
        if len(sub) < 3: return float("nan")
        x = np.arange(len(sub))
        return float(np.polyfit(x, sub[col].values, 1)[0])
    return sl(df.iloc[:t]), sl(df.iloc[t:2*t]), sl(df.iloc[2*t:])

def quartile_means(s):
    n = len(s); q = max(1, n // 4)
    return [s.iloc[i*q:min((i+1)*q, n)].mean() for i in range(4)]

def sign_label(v, pos_good=True):
    """Human-readable sign interpretation."""
    if np.isnan(v): return "n/a"
    s = f"{v:+.5f}/ep"
    if pos_good:
        return s + (" ↑ good" if v > 1e-6 else (" ↓ bad" if v < -1e-6 else " → flat"))
    else:
        return s + (" ↑ bad"  if v > 1e-6 else (" ↓ good" if v < -1e-6 else " → flat"))

def accel(s):
    """Second derivative (acceleration) — positive means slope is speeding up."""
    if len(s) < 10: return float("nan")
    n = len(s); t = n // 2
    s1 = slope_of(s.iloc[:t])
    s2 = slope_of(s.iloc[t:])
    return s2 - s1  # positive = improving faster in second half

def sep():
    print("=" * 72)

def subsep():
    print("-" * 72)

# ── Per-run text report ───────────────────────────────────────────────────────
all_reports = {}

for df, label in zip(datasets, labels):
    rep = {}
    all_reports[label] = rep

    disc_r_cols  = [c for c in df.columns if c.startswith("disc_reward_")]
    task_r_cols  = [c for c in df.columns if c.startswith("task_reward_")]
    score_r_cols = [c for c in df.columns if c.startswith("score_real_")]
    score_f_cols = [c for c in df.columns if c.startswith("score_fake_")]

    sep()
    print(f"\tRUN: {label}   (epochs {df.epoch.min():.0f}–{df.epoch.max():.0f})")
    sep()
    n_epochs = len(df)

    # ── SECTION 1: SURVIVAL ──────────────────────────────────────────────────
    lt = df["lifetime"]
    rep.update({
        "lt_mean":    lt.mean(),
        "lt_max":     lt.max(),
        "lt_final10": lt.tail(10).mean(),
        "lt_slope":   slope_of(lt),
        "lt_accel":   accel(lt),
        "lt_q":       quartile_means(lt),
    })
    es, em, el = phase_slopes(df, "lifetime")

    target   = CLIP_STEPS if CLIP_STEPS else EPISODE_LEN
    pct_clip = rep["lt_final10"] / target * 100
    pct_ep   = rep["lt_final10"] / EPISODE_LEN * 100

    print(f"\n[1] SURVIVAL  — lifetime = steps survived before fall/truncation per env")
    print(f"    Meaning: higher = agent balances longer. Target = motion clip ({target} steps/cycle) "
          f"then episode ({EPISODE_LEN} steps).")
    print(f"    mean={rep['lt_mean']:.1f}  max={rep['lt_max']:.1f}  "
          f"final-10-avg={rep['lt_final10']:.1f}  "
          f"({pct_clip:.0f}% of clip / {pct_ep:.0f}% of episode)")
    print(f"    quartiles (Q1→Q4): {[f'{v:.1f}' for v in rep['lt_q']]}")
    print(f"    overall slope : {sign_label(rep['lt_slope'], pos_good=True)}")
    print(f"    phase slopes  : early={es:+.4f}/ep  mid={em:+.4f}/ep  late={el:+.4f}/ep")
    print(f"    acceleration  : {rep['lt_accel']:+.4f}  "
          f"({'improving faster' if rep['lt_accel']>0 else 'slowing / regressing'} in second half)")

    if pct_ep < 10:
        diag = ("⚠️  CRITICAL – falls in <10% of episode. Very early termination. "
                "Physics or actuator issue likely. Check kp values, stiffness, contact threshold.")
    elif pct_clip < 50:
        diag = ("⚠️  POOR – surviving <50% of one motion cycle. Discriminator dominating. "
                "Policy needs more exploration or physics needs tuning.")
    elif pct_clip < 150:
        diag = ("⚡ PARTIAL – surviving 0.5–1.5 motion cycles. Imitation just starting. "
                "Expect slow steady improvement.")
    elif pct_ep < 80:
        diag = "✓  DECENT – partial episode survival, learning progressing."
    else:
        diag = "✓✓ GOOD – agent survives most of the episode."
    print(f"    → {diag}")

    # ── SECTION 2: DISCRIMINATOR ─────────────────────────────────────────────
    print(f"\n[2] DISCRIMINATOR  — real vs fake GAN scores")
    print(f"    Meaning: score_real = disc confidence on reference motion (want: stable ~0–0.3).")
    print(f"             score_fake = disc confidence on policy rollout (want: rising toward real → 0).")
    print(f"             gap = real−fake. Want → 0. Growing gap = disc winning. Shrinking gap = policy winning.")

    for sr_col, sf_col in zip(score_r_cols, score_f_cols):
        name = sr_col.replace("score_real_", "")
        sr = df[sr_col];  sf = df[sf_col]
        gap = sr - sf
        rep[f"sr_{name}_f10"]  = sr.tail(10).mean()
        rep[f"sf_{name}_f10"]  = sf.tail(10).mean()
        rep[f"gap_{name}_f10"] = gap.tail(10).mean()
        es_r, em_r, el_r = phase_slopes(df, sr_col)
        es_f, em_f, el_f = phase_slopes(df, sf_col)
        acc_f = accel(sf); acc_r = accel(sr)

        print(f"\n    [{name}]")
        print(f"      score_real : early={sr.head(5).mean():.3f} → final={sr.tail(10).mean():.3f}  "
              f"slope={sign_label(slope_of(sr), pos_good=False)}")
        print(f"                   phase: E={es_r:+.5f}  M={em_r:+.5f}  L={el_r:+.5f}  "
              f"accel={acc_r:+.4f}")
        print(f"        +ve slope = disc gets better at spotting real data (normal at first,")
        print(f"        then should stabilize. Continuously rising = disc diverging, no challenge left.)")
        print(f"      score_fake : early={sf.head(5).mean():.3f} → final={sf.tail(10).mean():.3f}  "
              f"slope={sign_label(slope_of(sf), pos_good=True)}")
        print(f"                   phase: E={es_f:+.5f}  M={em_f:+.5f}  L={el_f:+.5f}  "
              f"accel={acc_f:+.4f}")
        print(f"        +ve slope = policy fools disc more = GOOD. -ve = policy getting worse.")
        print(f"      gap (r-f)  : early={gap.head(5).mean():.3f} → final={gap.tail(10).mean():.3f}  "
              f"slope={sign_label(slope_of(gap), pos_good=False)}")
        print(f"        gap<0.05 = policy nearly matching reference. gap>0.5 = disc dominating.")

        gap_slope = slope_of(gap); fake_slope = slope_of(sf)
        if gap_slope > 0.0001 and fake_slope < 0.00005:
            print(f"      → ⚠️  DISC WINNING: gap growing AND fake flat. Policy cannot fool disc.")
        elif gap_slope > 0.0001 and fake_slope >= 0.00005:
            print(f"      → ⚡ DISC AHEAD but fake score is rising — policy IS learning, just slowly.")
            if gap.tail(100).mean() < gap.tail(10).mean():
                print(f"         Recent gap acceleration: may need more epochs before reversal.")
            else:
                print(f"         Gap appears to be stabilizing — the competition is finding equilibrium.")
        elif gap_slope <= 0:
            print(f"      → ✓  Gap stable or shrinking — policy is catching up to discriminator.")

    # ── SECTION 3: REWARD ───────────────────────────────────────────────────
    print(f"\n[3] REWARD DECOMPOSITION")
    print(f"    Meaning: disc_reward = log(disc_score_fake/(1−disc_score_fake)), used as RL reward.")
    print(f"             0 = policy perfectly fools disc. Negative = disc confident rollout is fake.")
    print(f"             Rising toward 0 = improvement. Falling away = disc pulling ahead.")
    print(f"    reward_mean: mean={df.reward_mean.mean():.4f}  final10={df.reward_mean.tail(10).mean():.4f}  "
          f"slope={sign_label(slope_of(df.reward_mean), pos_good=True)}")
    es_rw, em_rw, el_rw = phase_slopes(df, "reward_mean")
    print(f"    phase: E={es_rw:+.5f}  M={em_rw:+.5f}  L={el_rw:+.5f}")
    first_positive_ep = df[df.reward_mean > 0]
    if not first_positive_ep.empty:
        fp = first_positive_ep.iloc[0]
        print(f"    First +ve reward at epoch {int(fp.epoch)} (reward={fp.reward_mean:.4f})")
    for col in disc_r_cols:
        name = col.replace("disc_reward_","")
        rep[f"dr_{name}_f10"]  = df[col].tail(10).mean()
        rep[f"dr_{name}_slope"]= slope_of(df[col])
        print(f"    disc_reward [{name}]: mean={df[col].mean():.4f}  final10={df[col].tail(10).mean():.4f}  "
              f"slope={slope_of(df[col]):+.6f}/ep  accel={accel(df[col]):+.4f}")

    # ── SECTION 4: VALUE FUNCTION ─────────────────────────────────────────
    vl = df["value_loss"]
    rep["vl_f10"]  = vl.tail(10).mean()
    rep["vl_slope"]= slope_of(vl)
    print(f"\n[4] VALUE FUNCTION  — critic loss")
    print(f"    Meaning: value_loss = MSE between predicted V(s) and actual discounted returns.")
    print(f"             Falling = critic learning the actual reward landscape.")
    print(f"             Very low with negative disc_reward = critic learned 'falling is worth X'.")
    print(f"             Should settle, not oscillate. Large spikes = unstable returns.")
    print(f"    mean={vl.mean():.5f}  final10={rep['vl_f10']:.5f}  "
          f"slope={sign_label(rep['vl_slope'], pos_good=False)}")
    es_v, em_v, el_v = phase_slopes(df, "value_loss")
    print(f"    phase: E={es_v:+.5f}  M={em_v:+.5f}  L={el_v:+.5f}")
    if rep["vl_f10"] < 0.005:
        print(f"    → ✓  Near-zero: critic converged. Check whether reward itself is meaningful.")
    elif rep["vl_f10"] < 0.05:
        print(f"    → ✓  Low and decreasing: critic converging well.")
    elif rep["vl_f10"] > 1.0:
        print(f"    → ⚠️  HIGH (>1.0): critic is unstable. Consider reducing critic_lr or checking for reward spikes.")
    else:
        print(f"    → ⚡ Moderate value loss. Still converging — normal at this stage.")

    # ── SECTION 5: POLICY (PPO) ───────────────────────────────────────────
    pl = df["policy_loss"]
    rep["pl_early"]  = pl.head(20).mean()
    rep["pl_f10"]    = pl.tail(10).mean()
    rep["pl_slope"]  = slope_of(pl)
    print(f"\n[5] POLICY (PPO LOSS)")
    print(f"    Meaning: policy_loss = −advantage-weighted log-prob ratio (clipped).")
    print(f"             Starts near 0. Becomes more negative as PPO tightens the policy.")
    print(f"             Moderately negative (−0.01 to −0.05) = healthy updates.")
    print(f"             Very negative (<−0.1) = large updates; check if advantageous actions are correct.")
    print(f"             Plateau at 0 = no gradient, policy not being updated.")
    print(f"    early={rep['pl_early']:.4f}  final10={rep['pl_f10']:.4f}  "
          f"slope={sign_label(rep['pl_slope'], pos_good=False)}")
    es_p, em_p, el_p = phase_slopes(df, "policy_loss")
    print(f"    phase: E={es_p:+.5f}  M={em_p:+.5f}  L={el_p:+.5f}")
    if abs(rep["pl_f10"]) < 0.001:
        print(f"    → ⚠️  Near-zero: policy stalled. Consider entropy bonus or lr schedule.")
    elif rep["pl_f10"] < -0.1:
        print(f"    → ⚠️  Very negative: large updates. Watch for policy collapse or mode drift.")
    else:
        print(f"    → ✓  Moderate negative: healthy PPO behaviour.")

    # ── SECTION 6: TERMINATE INCENTIVE CHECK ────────────────────────────
    print(f"\n[6] TERMINATE-REWARD SAFETY CHECK")
    print(f"    Meaning: if terminate_reward > disc_r_long_horizon, the agent earns more by")
    print(f"    falling on purpose than by surviving with negative disc_reward.")
    print(f"    We check: adv(fall) = terminate_reward - V_long_horizon.  Positive = BAD.")
    if disc_r_cols:
        dr = df[disc_r_cols[0]]
        V_long = dr / (1 - GAMMA)
        adv_fall = TERMINATE_REWARD - V_long
        bad = df[adv_fall > 0]
        if not bad.empty:
            r0 = bad.iloc[0]
            print(f"    ⚠️  Falling first REWARDED at epoch {int(r0.epoch)} "
                  f"(disc_r={r0[disc_r_cols[0]]:.3f}, adv={adv_fall.loc[r0.name]:+.2f})")
            print(f"       terminate_reward={TERMINATE_REWARD} is too weak for this reward level.")
        else:
            print(f"    ✓  terminate_reward={TERMINATE_REWARD} keeps falling penalised throughout.")
        final_adv = adv_fall.iloc[-1]
        safe_tr   = V_long.min() - 2.0
        print(f"    adv(fall) at final epoch: {final_adv:+.2f} "
              f"({'⚠️ falling is incentivised' if final_adv > 0 else '✓ falling still penalised'})")
        if safe_tr < TERMINATE_REWARD:
            print(f"    → Recommended terminate_reward: {max(-20.0, safe_tr - 1):.0f} "
                  f"(formula: disc_r_worst/(1−γ) = {dr.min():.3f}/{1-GAMMA:.2f} = {dr.min()/(1-GAMMA):.1f})")

    # ── SECTION 7: PLATEAU / COLLAPSE ────────────────────────────────────
    print(f"\n[7] PLATEAU / COLLAPSE DETECTION (window={args.window})")
    lt_roll = lt.rolling(args.window, min_periods=1)
    lt_std   = lt_roll.std()
    lt_mean_roll = lt_roll.mean()
    collapse_mask = (lt_mean_roll.diff() < -0.5) & (lt_mean_roll < 15)
    n_coll = int(collapse_mask.sum())
    rep["n_collapse"] = n_coll
    last30_lt = df.iloc[int(n_epochs*0.7):]["lifetime"]
    lt_range_last30 = last30_lt.max() - last30_lt.min()
    rep["lt_range_l30"] = lt_range_last30
    print(f"    rolling std(lifetime): mean={lt_std.mean():.2f}  (higher = noisier training)")
    print(f"    collapse events (drop>0.5 while lt<15): {n_coll}")
    if lt_range_last30 < 1.0:
        print(f"    → ⚠️  STAGNATION: lifetime variation in last 30% = {lt_range_last30:.2f} (no progress).")
    else:
        print(f"    → Lifetime range in last 30%: {lt_range_last30:.2f}  "
              f"({'has variance, still active' if lt_range_last30 > 3 else 'stabilising'})")

# ── OVERALL SUMMARY ───────────────────────────────────────────────────────────

sep()
print(f"\tNARRATIVE SUMMARY")
sep()

for label, rep in all_reports.items():
    df   = datasets[labels.index(label)]
    n    = len(df)
    target = CLIP_STEPS if CLIP_STEPS else EPISODE_LEN
    pct_clip = rep["lt_final10"] / target * 100
    pct_ep   = rep["lt_final10"] / EPISODE_LEN * 100
    disc_r_f10_keys = [k for k in rep if k.startswith("dr_") and k.endswith("_f10")]
    disc_r_latest = rep[disc_r_f10_keys[0]] if disc_r_f10_keys else float("nan")
    disc_r_slope_keys = [k for k in rep if k.startswith("dr_") and k.endswith("_slope")]
    disc_r_slope = rep[disc_r_slope_keys[0]] if disc_r_slope_keys else float("nan")

    print(f"\n[{label}]")
    print(f"  Epochs trained : {df.epoch.min():.0f} – {df.epoch.max():.0f}")
    print(f"  Lifetime       : {rep['lt_final10']:.1f} steps  ({pct_clip:.0f}% of clip / {pct_ep:.0f}% of episode)  "
          f"slope={rep['lt_slope']:+.4f}/ep  accel={rep['lt_accel']:+.4f}")
    print(f"  Disc reward    : {disc_r_latest:.4f}  slope={disc_r_slope:+.6f}/ep  "
          f"({'rising ✓' if disc_r_slope > 0 else 'falling ⚠️'})")
    print(f"  Value loss     : {rep['vl_f10']:.5f}  slope={rep['vl_slope']:+.6f}/ep")
    print(f"  Collapse events: {rep['n_collapse']}")

    # Narrative
    print()
    issues = []
    positives = []

    if pct_ep < 10:
        issues.append("  CRITICAL: Lifetime is <10% of episode — physics or reward fundamentally broken.")
    elif pct_clip < 50:
        issues.append("  Lifetime is <50% of one motion cycle — policy barely holding posture.")
    else:
        positives.append("  Lifetime is adequate — posture is partially maintained.")

    # Discriminator gap trend
    gap_keys = [k for k in rep if k.startswith("gap_") and k.endswith("_f10")]
    sf_keys  = [k for k in rep if k.startswith("sf_") and k.endswith("_f10")]
    if gap_keys and sf_keys:
        gap_f10 = rep[gap_keys[0]]
        sf_f10  = rep[sf_keys[0]]
        sf_slope_k = gap_keys[0].replace("gap_","sf_").replace("_f10","_slope") \
                     if gap_keys[0].replace("gap_","sf_").replace("_f10","_slope") in rep else None
        if sf_f10 > -0.01:
            positives.append(f"  score_fake={sf_f10:.3f} is approaching real — policy IS fooling the disc.")
        elif sf_f10 > -0.05:
            positives.append(f"  score_fake={sf_f10:.3f} trending up — slow but real progress.")
        else:
            issues.append(f"  score_fake={sf_f10:.3f} still very negative — policy not close to reference yet.")
        if gap_f10 < 0.1:
            positives.append(f"  Gap={gap_f10:.3f} — disc and policy nearly converged. Near-perfect imitation.")
        elif gap_f10 < 0.35:
            positives.append(f"  Gap={gap_f10:.3f} — stabilising. Disc ahead but policy is competing.")
        else:
            issues.append(f"  Gap={gap_f10:.3f} — discriminator still dominating.")

    if disc_r_slope > 0:
        positives.append(f"  disc_reward is rising (+{disc_r_slope:.5f}/ep) — imitation quality improving over time.")
    else:
        issues.append(f"  disc_reward trending down — policy losing ground to discriminator.")

    if rep["vl_f10"] < 0.05:
        positives.append(f"  Value loss={rep['vl_f10']:.5f} — critic stable.")
    else:
        issues.append(f"  Value loss={rep['vl_f10']:.5f} — critic still converging.")

    for p in positives:
        print(f"  ✓ {p.strip()}")
    for i in issues:
        print(f"  ⚠ {i.strip()}")

# ── PLOTTING ──────────────────────────────────────────────────────────────────
W = args.window
fig = plt.figure(figsize=(20, 16))
fig.suptitle("ICCGAN Training Diagnostics", fontsize=13, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(4, 3, hspace=0.52, wspace=0.38,
                       left=0.06, right=0.97, top=0.95, bottom=0.05)

ax_lt   = fig.add_subplot(gs[0, :2])
ax_lt_z = fig.add_subplot(gs[0, 2])   # zoomed lifetime
ax_disc = fig.add_subplot(gs[1, :2])
ax_fake = fig.add_subplot(gs[1, 2])
ax_dr   = fig.add_subplot(gs[2, :2])
ax_dr_z = fig.add_subplot(gs[2, 2])   # disc_reward zoomed last 30%
ax_vl   = fig.add_subplot(gs[3, 0])
ax_pl   = fig.add_subplot(gs[3, 1])
ax_gap  = fig.add_subplot(gs[3, 2])

colors = plt.cm.tab10.colors

for i, (df, label) in enumerate(zip(datasets, labels)):
    c  = colors[i % len(colors)]
    ep = df["epoch"]
    n  = len(df)

    dr_col = next((x for x in df.columns if x.startswith("disc_reward_")), None)
    sr_col = next((x for x in df.columns if x.startswith("score_real_")),  None)
    sf_col = next((x for x in df.columns if x.startswith("score_fake_")),  None)

    # ── Lifetime (full)
    lt_s = smooth(df.lifetime, W)
    ax_lt.plot(ep, lt_s, label=label, color=c, lw=1.5)
    ax_lt.fill_between(ep,
        df.lifetime.rolling(W, min_periods=1).quantile(0.1),
        df.lifetime.rolling(W, min_periods=1).quantile(0.9),
        alpha=0.12, color=c)
    if CLIP_STEPS:
        ax_lt.axhline(CLIP_STEPS, color=c, linestyle=":", lw=1, alpha=0.6)

    # ── Lifetime zoomed (last 30%)
    cut = int(n * 0.7)
    ax_lt_z.plot(ep.iloc[cut:], lt_s.iloc[cut:], color=c, lw=1.5, label=label)
    ax_lt_z.fill_between(ep.iloc[cut:],
        df.lifetime.iloc[cut:].rolling(W, min_periods=1).quantile(0.1),
        df.lifetime.iloc[cut:].rolling(W, min_periods=1).quantile(0.9),
        alpha=0.15, color=c)
    if CLIP_STEPS:
        ax_lt_z.axhline(CLIP_STEPS, color="gray", linestyle=":", lw=1)

    # ── Discriminator scores
    if sr_col and sf_col:
        ax_disc.plot(ep, smooth(df[sr_col], W), label=f"{label} real", color=c, lw=1.5)
        ax_disc.plot(ep, smooth(df[sf_col], W), label=f"{label} fake", color=c,
                     lw=1.5, linestyle="--")
        gap = df[sr_col] - df[sf_col]
        ax_gap.plot(ep, smooth(gap, W), color=c, label=label, lw=1.5)
        ax_fake.plot(ep, smooth(df[sf_col], W), color=c, label=label, lw=1.5)

    # ── Disc reward
    if dr_col:
        ax_dr.plot(ep, smooth(df[dr_col], W), color=c, label=label, lw=1.5)
        # Zoomed last 30%
        ax_dr_z.plot(ep.iloc[cut:], smooth(df[dr_col], W).iloc[cut:],
                     color=c, label=label, lw=1.5)
        ax_dr_z.fill_between(ep.iloc[cut:],
            df[dr_col].iloc[cut:].rolling(W, min_periods=1).quantile(0.1),
            df[dr_col].iloc[cut:].rolling(W, min_periods=1).quantile(0.9),
            alpha=0.12, color=c)

    # ── Value loss (log)
    vl_cl = df.value_loss.clip(lower=1e-6)
    ax_vl.semilogy(ep, vl_cl, alpha=0.2, color=c)
    ax_vl.semilogy(ep, smooth(vl_cl, W), color=c, label=label, lw=1.5)

    # ── Policy loss
    ax_pl.plot(ep, smooth(df.policy_loss, W), color=c, label=label, lw=1.5)

# ── Formatting helper
def fmt(ax, title, ylabel="", hline=None, note=None, legend=True):
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
    ax.set_xlabel("Epoch", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.18)
    if legend: ax.legend(fontsize=6, loc="best")
    if hline is not None:
        ax.axhline(hline, color="gray", linestyle=":", lw=0.8)
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes,
                fontsize=5.5, va="top", color="#444444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

target_label = f"clip={CLIP_STEPS}steps" if CLIP_STEPS else f"ep={EPISODE_LEN}"
fmt(ax_lt,   f"Lifetime  (want → {target_label})", "Steps",
    note=f"Dotted = motion clip ({CLIP_STEPS} steps)" if CLIP_STEPS else None)
fmt(ax_lt_z, "Lifetime  (last 30% zoomed)", "Steps")
fmt(ax_disc, "Disc Scores: Real (solid) vs Fake (dashed)", "Score", hline=0,
    note="Want: real≈fake≈0  |  gap shrinking")
fmt(ax_fake, "Fake Score only\n(+ve = policy fooling disc)", "Score", hline=0,
    note="Rising toward 0 = learning")
fmt(ax_dr,   "Disc Reward  (want → 0)", "Reward", hline=0,
    note="Negative = disc winning  |  Rising = policy improving")
fmt(ax_dr_z, "Disc Reward  (last 30% zoomed)", "Reward", hline=0)
fmt(ax_vl,   "Value Loss  (log, want ↓)", "Loss")
fmt(ax_pl,   "Policy Loss (PPO)\nModerately −ve = healthy", "Loss", hline=0,
    note="−0.01–−0.05 healthy  |  >0 or <<−0.1 bad")
fmt(ax_gap,  "Score Gap = real−fake\n(want → 0)", "Gap", hline=0,
    note="Shrinking = policy winning  |  >0.5 = disc dominating")

# ── Healthy targets box
targets_text = (
    "HEALTHY TARGETS\n"
    "─────────────────────────\n"
    f"lifetime  → {target_label}\n"
    "score_fake → 0  (fools disc)\n"
    "gap       → 0  (converged)\n"
    "disc_reward → 0  (imitation perfect)\n"
    "value_loss → near 0\n"
    "policy_loss: −0.01…−0.05"
)
ax_ann = fig.add_axes([0.005, 0.005, 0.22, 0.12])
ax_ann.axis("off")
ax_ann.text(0.0, 0.5, targets_text, fontsize=6, va="center", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

# ── Save
out_path = os.path.join(args.out, "training_diagnostics.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\n→ Plot saved: {out_path}")

sep()
print("REFERENCE: metric sign meanings at a glance")
sep()
print("""
 Metric             | +ve slope means          | −ve slope means
 ───────────────────┼──────────────────────────┼──────────────────────
 lifetime           | surviving longer  ✓       | falling sooner  ✗
 score_fake         | policy fools disc ✓       | disc winning    ✗
 score_real         | disc better at real ✓/✗   | disc losing real signal
 gap (real−fake)    | disc pulling ahead ✗      | policy catching up ✓
 disc_reward        | imitation improving ✓     | disc pulling ahead ✗
 value_loss         | critic diverging   ✗      | critic converging ✓
 policy_loss        | → 0 = policy stalled ✗   | moderate −ve = updates ✓
──────────────────────────────────────────────────────────────────────
 disc_reward interpretation:
   +ve  → policy currently fooling disc (near-perfect imitation)
   ~0   → competitive balance (healthy mid-training state)
   −0.1 → policy recognisably different from reference
   −0.5 → policy very far from reference motion
""")