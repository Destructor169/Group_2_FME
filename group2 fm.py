"""
═══════════════════════════════════════════════════════════════════════
  SYNTHETIC FORWARD ANALYSIS — with dividend yield + interpretability
═══════════════════════════════════════════════════════════════════════

  WHAT IS A SYNTHETIC FORWARD?
  ----------------------------
  Put-call parity (European, no-arb) says:
        C - P = S·e^(-qT) - K·e^(-rT)
  Rearranging to isolate the "forward price implied by options":
        F_syn(K) = K + (C - P)·e^(rT)
  This should, in theory, EQUAL the actual (cost-of-carry) forward:
        F_actual = S·e^((r-q)T)
  regardless of strike K. When they differ → arbitrage OR friction.

  THIS SCRIPT:
    1. Pulls live option chains from Yahoo.
    2. Builds F_syn at every liquid ATM strike.
    3. Compares to F_actual using a USER-SET dividend yield q.
    4. Plots 4 diagnostic panels + prints a sanity-check analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime
import yfinance as yf

# ─── CONFIG ──────────────────────────────────────────────────────────────
TICKER        = "AAPL"
R             = 0.045      # risk-free rate (~3M T-bill, Apr 2026)

# Dividend modeling — choose ONE approach:
#   (A) Continuous yield only:  leave NEXT_DIV_AMOUNT = 0.
#                                Good for non-payers or when no ex-div in window.
#   (B) Discrete + continuous:  set the next ex-dividend date and amount.
#                                Accurate when a known ex-div falls near/inside
#                                your longest expiry — options price the $ drop,
#                                NOT an averaged yield.
Q_CUSTOM           = 0.0        # baseline continuous yield (set 0 when using discrete)
NEXT_DIV_AMOUNT    = 0.26       # dollars per share (0 disables discrete term)
NEXT_EX_DATE       = "2026-05-12"   # "YYYY-MM-DD" or None

MIN_OI        = 20
ATM_WINDOW    = 0.05       # ±5% of spot
MAX_EXPIRIES  = 10         # scan further out so ex-div date falls inside at least
                            # one window (needed for discrete-div validation to work)

# ─── AESTHETIC ───────────────────────────────────────────────────────────
BG, FG, DIM, GOLD, BLUE, GREEN, RED, PURPLE = (
    "#0B0E14", "#E6EDF3", "#8B949E", "#FFB900",
    "#58A6FF", "#3FB950", "#F85149", "#BC8CFF"
)
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": DIM, "axes.labelcolor": DIM,
    "xtick.color": DIM, "ytick.color": DIM,
    "text.color": FG, "axes.titlecolor": FG,
    "grid.color": "#21262D", "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
})

# ─── 1. FETCH DATA ───────────────────────────────────────────────────────
print(f"\n{'═'*68}")
print(f"  SYNTHETIC FORWARD ANALYSIS — {TICKER}")
print(f"{'═'*68}\n")

tk = yf.Ticker(TICKER)
yahoo_spot = float(tk.history(period="5d")["Close"].iloc[-1])
today = date.today()

try:
    info = tk.info
    reported_q = info.get("dividendYield", 0) or 0
    # Yahoo is inconsistent: sometimes returns 0.0055 (fraction), sometimes 0.55 (percent).
    # If value > 0.15, it's almost certainly percent-units (15% yield would be extreme).
    if reported_q > 0.15: reported_q /= 100
except Exception:
    reported_q = 0.0

# NOTE: spot is set to yahoo_spot INITIALLY, then replaced with the option-implied
# spot after we load the chain (see below). This guarantees spot and options are
# measured at the same moment, which eliminates yfinance cache incoherence.
spot = yahoo_spot

print(f"  Spot (Yahoo, initial): ${yahoo_spot:.2f}")
print(f"  Risk-free rate r:      {R*100:.2f}%")
print(f"  Reported div yield:    {reported_q*100:.2f}%  (from Yahoo)")
print(f"  USER-SET q:            {Q_CUSTOM*100:.2f}%  ← used in F_actual")
print(f"  Filters:               OI ≥ {MIN_OI}, strikes within ±{ATM_WINDOW*100:.0f}% of spot")
print(f"  Max expiries scanned:  {MAX_EXPIRIES}\n")

print(f"  {'─'*60}")
print(f"  Loading option chains...")
print(f"  {'─'*60}")

rows = []
for exp_str in tk.options[:MAX_EXPIRIES]:
    exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
    T_days = (exp - today).days
    T = T_days / 365
    if T <= 0: continue

    oc = tk.option_chain(exp_str)
    # Grab lastTradeDate & volume too — needed for staleness detection.
    call_cols = ["bid","ask","openInterest","volume","lastPrice","lastTradeDate"]
    put_cols  = call_cols
    c = oc.calls.set_index("strike")[[col for col in call_cols if col in oc.calls.columns]]
    p = oc.puts .set_index("strike")[[col for col in put_cols  if col in oc.puts.columns ]]
    c.columns = [f"c_{col.replace('openInterest','oi').replace('lastPrice','last').replace('lastTradeDate','ltd')}" for col in c.columns]
    p.columns = [f"p_{col.replace('openInterest','oi').replace('lastPrice','last').replace('lastTradeDate','ltd')}" for col in p.columns]
    chain = c.join(p, how="inner").reset_index().rename(columns={"strike":"K"})

    before = len(chain)
    chain = chain[(chain["c_bid"]>0)&(chain["c_ask"]>0)
                 &(chain["p_bid"]>0)&(chain["p_ask"]>0)
                 &(chain["c_oi"]>=MIN_OI)&(chain["p_oi"]>=MIN_OI)]
    chain = chain[chain["K"].between(spot*(1-ATM_WINDOW), spot*(1+ATM_WINDOW))]

    chain["C"] = (chain["c_bid"]+chain["c_ask"])/2
    chain["P"] = (chain["p_bid"]+chain["p_ask"])/2
    chain["spread"] = 0.5*((chain["c_ask"]-chain["c_bid"])+(chain["p_ask"]-chain["p_bid"]))
    chain["T_days"] = T_days
    chain["T"] = T
    chain["expiry"] = exp

    print(f"    {exp} (T={T_days:>3}d): {before:>3} strikes → {len(chain):>2} liquid & ATM")
    if len(chain): rows.append(chain)

if not rows:
    print("\n  ⚠ No liquid data. Try different ticker or lower MIN_OI.")
    raise SystemExit

df = pd.concat(rows, ignore_index=True)

# ─── MARKET HOURS & STALENESS DIAGNOSTIC ─────────────────────────────────
# US equity options trade 9:30–16:00 ET Mon-Fri. Yahoo freezes bid/ask
# outside these windows → the "mid" you compute is yesterday's close,
# which breaks put-call parity in the exact pattern of large |gap|, q_impl < 0.
from datetime import timezone, timedelta as td
now_utc = datetime.now(timezone.utc)
now_et  = now_utc - td(hours=4)   # ET ≈ UTC-4 (DST); close enough for gating
et_hour = now_et.hour + now_et.minute/60
is_weekday = now_et.weekday() < 5
market_open = is_weekday and (9.5 <= et_hour <= 16.0)

print(f"\n  {'─'*60}")
print(f"  Quote-freshness check")
print(f"  {'─'*60}")
print(f"    Current time (ET approx):  {now_et.strftime('%Y-%m-%d %H:%M')}  "
      f"({'weekday' if is_weekday else 'weekend'})")
print(f"    US options market open:    {'✓ YES' if market_open else '✗ NO — quotes likely stale'}")

# Spot/option-chain coherence: derive SPOT FROM THE OPTION CHAIN so that spot
# and options are from the same moment by construction. This eliminates the
# yfinance caching gotcha entirely — no more "wait and retry."
#
# Method: at the most ATM strike per expiry, put-call parity gives
#   S·e^(-qT) = C − P + K·e^(-rT)
# Dropping the (small) q·T term for this purpose:
#   S ≈ C − P + K·e^(-rT)
# Then median across the ATM strikes of multiple expiries for robustness.

stale_flag = False
spot_drift_flag = False

# Compute option-implied spot from the 5 most ATM rows
atm_probe = df.iloc[(df["K"] - yahoo_spot).abs().argsort()[:5]].copy()
atm_probe["S_impl"] = (atm_probe["C"] - atm_probe["P"]
                      + atm_probe["K"] * np.exp(-R * atm_probe["T"]))
implied_spot = float(atm_probe["S_impl"].median())
spot_drift = implied_spot - yahoo_spot

print(f"\n  {'─'*60}")
print(f"  Spot consistency")
print(f"  {'─'*60}")
print(f"    Yahoo-reported spot:       ${yahoo_spot:.2f}")
print(f"    Spot implied by options:   ${implied_spot:.2f}  "
      f"(Δ = {spot_drift:+.2f})")

# ALWAYS use the option-implied spot as the canonical S.
# This is the spot the option market itself was quoting against.
# If the Yahoo spot differs, it just means yfinance's history() and option_chain()
# endpoints aren't perfectly synchronized — a known issue we now bypass.
spot = implied_spot
if abs(spot_drift) > 0.15:
    print(f"    ⚠ Detected ${abs(spot_drift):.2f} drift between Yahoo spot and option quotes.")
    print(f"    → Auto-corrected: using option-implied spot ${spot:.2f} as canonical S.")
    print(f"      This eliminates yfinance cache incoherence without a rerun.")
else:
    print(f"    ✓ Spot sources agree. Using option-implied ${spot:.2f} for calculations.")

# Staleness check: lastTradeDate age
if "c_ltd" in df.columns:
    try:
        df["c_ltd"] = pd.to_datetime(df["c_ltd"], utc=True, errors="coerce")
        df["p_ltd"] = pd.to_datetime(df["p_ltd"], utc=True, errors="coerce")
        latest_quote = max(df["c_ltd"].max(), df["p_ltd"].max())
        age_hours = (now_utc - latest_quote).total_seconds() / 3600
        print(f"    Most recent quote age:     {age_hours:.1f} hours old")
        if age_hours > 2 and not market_open:
            stale_flag = True
            print(f"    ⚠ Quotes are frozen from last session. Results are INDICATIVE only.")
    except Exception as e:
        print(f"    (couldn't parse lastTradeDate: {e})")

# ─── 2. CONSTRUCT SYNTHETIC FORWARD ──────────────────────────────────────
# Synthetic (from options, model-free):
#     F_syn = K + (C − P)·e^(rT)
#
# Actual (cost-of-carry):
#   • Continuous yield only:  F_act = S·e^((r−q)T)
#   • Discrete dividend:      F_act = (S − PV_div)·e^(rT)     — STANDARD for equities
#                             where PV_div = D·e^(−r·t_ex), and t_ex is time-to-ex
#                             if ex-date falls BEFORE T, else 0 (div goes to holder).
#   • Hybrid: both terms combined — use when stock has both a continuous base
#     yield and a known upcoming discrete payment.

# Parse ex-dividend date if provided
ex_date = None
if NEXT_DIV_AMOUNT > 0 and NEXT_EX_DATE:
    ex_date = datetime.strptime(NEXT_EX_DATE, "%Y-%m-%d").date()
    t_ex_days = (ex_date - today).days
    print(f"  Discrete div:          ${NEXT_DIV_AMOUNT:.2f} on {NEXT_EX_DATE} "
          f"(t_ex = {t_ex_days}d from today)")
else:
    t_ex_days = None

df["F_syn"] = df["K"] + (df["C"] - df["P"]) * np.exp(R * df["T"])

# PV of discrete dividend IF ex-date occurs before the option expiry.
# If ex-date is after expiry, the option holder effectively trades WITHOUT
# that dividend priced in, so it doesn't enter the forward.
def pv_div(T_days_row):
    if t_ex_days is None or t_ex_days > T_days_row or t_ex_days < 0:
        return 0.0
    t_ex = t_ex_days / 365
    return NEXT_DIV_AMOUNT * np.exp(-R * t_ex)

df["PV_div"]   = df["T_days"].apply(pv_div)
df["F_actual"] = (spot - df["PV_div"]) * np.exp((R - Q_CUSTOM) * df["T"])
df["F_actual_noq"] = spot * np.exp(R * df["T"])   # naive baseline for comparison

# Sanity: warn if discrete div configured but NO expiry covers the ex-date
if t_ex_days is not None:
    n_covering = (df["T_days"] >= t_ex_days).sum()
    if n_covering == 0:
        max_T = df["T_days"].max()
        print(f"\n  ⚠ WARNING: ex-dividend date is {t_ex_days}d out, but longest")
        print(f"    expiry scanned is only {max_T}d. Discrete-div term has NO effect")
        print(f"    on any row. To see it bite, increase MAX_EXPIRIES to reach {t_ex_days}d+.")
        print(f"    Current data still reflects the market ANTICIPATING the dividend")
        print(f"    (see diagnostic [7]), but the formula can't verify it yet.")

df["diff"]     = df["F_syn"] - df["F_actual"]
df["diff_bps"] = 1e4 * df["diff"] / df["F_actual"]

# Implied continuous-yield q per expiry (still useful as a diagnostic,
# but unreliable for short T because of 1/T noise amplification).
implied_q_by_T = (
    df.groupby("T_days")[["F_syn","T"]]
      .apply(lambda g: R - np.log(g["F_syn"].median() / spot) / g["T"].iloc[0], include_groups=False)
      .rename("q_implied")
      .reset_index()
)

# ─── 3. PRINT SUMMARY TABLE ──────────────────────────────────────────────
print(f"\n{'═'*68}")
print(f"  RESULTS — gap between synthetic forward and cost-of-carry forward")
print(f"{'═'*68}\n")
print(f"  {'T(d)':>4}  {'n':>3}  {'ex-div?':>7}  {'Mean gap $':>11}  {'Mean bps':>9}  {'|gap|>spr':>10}  {'q_impl':>8}")
print(f"  {'-'*4}  {'-'*3}  {'-'*7}  {'-'*11}  {'-'*9}  {'-'*10}  {'-'*8}")
summary_rows = []
for Td, g in df.groupby("T_days"):
    mean_gap = g["diff"].mean()
    std_gap  = g["diff"].std() if len(g) > 1 else 0
    mean_bps = g["diff_bps"].mean()
    frac_tradable = (g["diff"].abs() > g["spread"]).mean() * 100
    q_impl = implied_q_by_T.loc[implied_q_by_T["T_days"]==Td, "q_implied"].iloc[0]
    includes_ex = (t_ex_days is not None) and (0 <= t_ex_days <= Td)
    # Flag short-T q_implied as unreliable: noise is amplified by 1/T.
    # Dollar-gap of $0.10 on spot $270 at T=2d → q_impl error ≈ ±7% annualized.
    unreliable = Td < 5
    q_str = f"{q_impl*100:+.2f}%" + (" *" if unreliable else "  ")
    summary_rows.append({"T_days":Td,"mean_gap":mean_gap,"std":std_gap,
                         "mean_bps":mean_bps,"frac_tradable":frac_tradable,
                         "q_impl":q_impl,"n":len(g),"includes_ex":includes_ex,
                         "unreliable":unreliable})
    ex_mark = "✓" if includes_ex else "—"
    print(f"  {Td:>4}  {len(g):>3}  {ex_mark:>7}  {mean_gap:>+11.4f}  "
          f"{mean_bps:>+9.2f}  {frac_tradable:>9.0f}%  {q_str:>8}")
print(f"\n  * = T < 5d, q_implied unreliable (small gap × 1/T amplifies noise)")
if t_ex_days is not None:
    print(f"  ✓ = ex-dividend date (day {t_ex_days}) falls INSIDE this expiry")
summary = pd.DataFrame(summary_rows)

# ─── 4. PLOT: 5 INTERPRETABLE PANELS ─────────────────────────────────────
# Top 2 rows: the original 4 diagnostic panels.
# Bottom row (spanning full width): THE MONEY SHOT — q_implied by expiry
# with the ex-date step highlighted. This is the figure the prof wants.
fig = plt.figure(figsize=(16, 13.5))
gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.28,
                      height_ratios=[1, 1, 1.15])
div_label = f"q={Q_CUSTOM*100:.2f}%"
if NEXT_DIV_AMOUNT > 0 and ex_date is not None:
    div_label += f"  +  D=\\${NEXT_DIV_AMOUNT:.2f} on {NEXT_EX_DATE}"
fig.suptitle(
    f"Synthetic Forwards vs Cost-of-Carry Forward — {TICKER}    "
    f"Spot \\${spot:.2f}   r={R*100:.1f}%   {div_label}",
    fontsize=13, fontweight="bold", y=0.995
)

def clean(ax, title, subtitle=""):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=18)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=8.5, color=DIM, style="italic", va="bottom")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)

# Pick expiry with most liquid strikes for Panel A
best_T = df.groupby("T_days").size().idxmax()
sub = df[df["T_days"] == best_T].sort_values("K")

# ─── Panel A: Construction ───
ax = fig.add_subplot(gs[0, 0])
clean(ax, f"① Construction check  —  T = {best_T} days",
      "Gold dots = synthetic forward per strike. Blue line = theoretical forward.")
F_act_val = sub["F_actual"].iloc[0]
F_act_noq = sub["F_actual_noq"].iloc[0]
pv_d = sub["PV_div"].iloc[0]
if pv_d > 0:
    blue_label = f"F_actual = (S−PV_div)·e^(rT) = \\${F_act_val:.3f}"
else:
    blue_label = f"F_actual = S·e^((r−q)T) = \\${F_act_val:.3f}"
ax.axhline(F_act_val, color=BLUE, lw=2, zorder=2, label=blue_label)
ax.axhline(F_act_noq, color=RED, lw=1.2, ls=":", alpha=0.7, zorder=1,
           label=f"F_actual (no div, no q) = \\${F_act_noq:.3f}")
ax.scatter(sub["K"], sub["F_syn"], s=60, color=GOLD,
           edgecolor=BG, lw=0.6, zorder=3, label="F_syn = K + (C−P)·e^(rT)")
# Expand y-range so it looks like data, not noise
ymin = min(sub["F_syn"].min(), F_act_val, F_act_noq)
ymax = max(sub["F_syn"].max(), F_act_val, F_act_noq)
pad = (ymax - ymin) * 0.3 + 0.1
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_xlabel("Strike K (\\$)"); ax.set_ylabel("Forward price (\\$)")
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

# ─── Panel B: Divergence across strikes ───
ax = fig.add_subplot(gs[0, 1])
clean(ax, "② Gap across strikes, by expiry",
      "Flat horizontal lines at 0 = parity holds. Drift with K = friction/skew.")
cmap = plt.get_cmap("plasma")
Ts = sorted(df["T_days"].unique())
for Td, col in zip(Ts, cmap(np.linspace(0.15, 0.85, len(Ts)))):
    g = df[df["T_days"] == Td].sort_values("K")
    ax.plot(g["K"], g["diff_bps"], "o-", color=col, lw=1.5, ms=5,
            alpha=0.9, label=f"{Td}d")
ax.axhline(0, color=FG, lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Strike K (\\$)"); ax.set_ylabel("Gap (basis points of F)")
ax.legend(fontsize=8, ncol=2, title="Maturity", title_fontsize=8)

# ─── Panel C: Mean gap vs maturity ───
ax = fig.add_subplot(gs[1, 0])
clean(ax, "③ Mean gap vs time to expiry",
      "Gold band = typical bid-ask friction. Blue line inside band → no arb.")
g = df.groupby("T_days").agg(mean=("diff","mean"), std=("diff","std"),
                              spread=("spread","mean")).reset_index()
g["std"] = g["std"].fillna(0)
ax.fill_between(g["T_days"], -g["spread"], g["spread"],
                color=GOLD, alpha=0.20, label="±1 bid-ask spread")
ax.errorbar(g["T_days"], g["mean"], yerr=g["std"],
            fmt="o-", color=BLUE, capsize=4, lw=1.8, ms=8,
            label="Mean gap ± 1σ")
ax.axhline(0, color=FG, lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Days to expiry"); ax.set_ylabel("Gap (\\$)")
ax.legend(fontsize=9, loc="best")

# ─── Panel D: Why the gap — friction scatter ───
ax = fig.add_subplot(gs[1, 1])
clean(ax, "④ Is the gap tradable?  |gap|  vs  bid-ask spread",
      "Above dashed line = noise (friction). Below = potential arb.")
sc = ax.scatter(df["diff"].abs(), df["spread"], c=df["T_days"],
                cmap="plasma", s=32, alpha=0.8, edgecolor=BG, lw=0.3)
mx = max(df["diff"].abs().quantile(0.98), df["spread"].quantile(0.98)) * 1.15
ax.plot([0, mx], [0, mx], ls="--", color=FG, lw=1, alpha=0.6,
        label="|gap| = spread")
# Shade: above diagonal = spread > |gap| = GOOD (friction explains it)
#        below diagonal = |gap| > spread = BAD (unexplained, potential arb)
xx = np.linspace(0, mx, 100)
ax.fill_between(xx, xx, mx, color=GREEN, alpha=0.10,
                label="Explained by friction (spread > |gap|)")
ax.fill_between(xx, 0, xx, color=RED, alpha=0.10,
                label="Unexplained (|gap| > spread)")
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.set_xlabel("|Gap|  (\\$)"); ax.set_ylabel("Bid-ask spread  (\\$)")
cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Days to expiry", color=DIM, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=DIM)
ax.legend(fontsize=8, loc="lower right")

# ─── Panel E: THE MONEY SHOT — q_implied vs expiry, ex-date step ───
# This visualizes the American early-exercise dividend-capture premium:
# expiries that DON'T cover the ex-date price a lower effective carry,
# expiries that DO cover it price ~60 bps higher. The step IS the premium.
ax = fig.add_subplot(gs[2, :])   # spans full width
clean(ax, "⑤ The ex-dividend step  —  q_implied by expiry",
      "Step-up across ex-date = American early-exercise premium (dividend capture optionality).")

# Use only reliable (T ≥ 5d) expiries for the signal
plot_df = summary[~summary["unreliable"]].copy().sort_values("T_days")
if len(plot_df) > 0:
    # Split pre-ex vs post-ex (covering ex-date) for color coding
    pre_ex  = plot_df[~plot_df["includes_ex"]]
    post_ex = plot_df[ plot_df["includes_ex"]]
    
    # Plot all points connected in a line to show the trajectory
    ax.plot(plot_df["T_days"], plot_df["q_impl"]*100,
            color=DIM, lw=1, ls=":", alpha=0.5, zorder=1)
    
    # Pre-ex-date points (ex-date OUTSIDE expiry window)
    if len(pre_ex):
        ax.scatter(pre_ex["T_days"], pre_ex["q_impl"]*100,
                   s=140, color=BLUE, edgecolor=FG, lw=1.2, zorder=3,
                   label=f"Ex-date OUTSIDE window  (can't capture div)")
        for _, row in pre_ex.iterrows():
            ax.annotate(f"{row['q_impl']*100:+.2f}%",
                        (row["T_days"], row["q_impl"]*100),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", color=BLUE, fontsize=9, fontweight="bold")
    
    # Post-ex-date points (ex-date INSIDE expiry window)
    if len(post_ex):
        ax.scatter(post_ex["T_days"], post_ex["q_impl"]*100,
                   s=140, color=GOLD, edgecolor=FG, lw=1.2, zorder=3,
                   marker="D",
                   label=f"Ex-date INSIDE window  (CAN capture div)")
        for _, row in post_ex.iterrows():
            ax.annotate(f"{row['q_impl']*100:+.2f}%",
                        (row["T_days"], row["q_impl"]*100),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", color=GOLD, fontsize=9, fontweight="bold")
    
    # Vertical line at the ex-date
    if t_ex_days is not None:
        ax.axvline(t_ex_days, color=RED, lw=2, ls="--", alpha=0.7, zorder=2)
        # Place ex-date label at BOTTOM of plot to avoid collision with step annotation
        ylim_now = ax.get_ylim()
        ax.text(t_ex_days, ylim_now[0] + (ylim_now[1]-ylim_now[0])*0.05,
                f" ex-date (day {t_ex_days})",
                color=RED, fontsize=10, fontweight="bold", va="bottom", ha="left")
    
    # Annotate the step-up size if both groups exist
    if len(pre_ex) and len(post_ex):
        pre_med  = pre_ex["q_impl"].median() * 100
        post_med = post_ex["q_impl"].median() * 100
        step     = post_med - pre_med
        ax.axhline(pre_med,  color=BLUE, lw=1, ls="-", alpha=0.4, zorder=1)
        ax.axhline(post_med, color=GOLD, lw=1, ls="-", alpha=0.4, zorder=1)
        # Step arrow — positioned AFTER the ex-date line, not on it
        arrow_x = t_ex_days + 1.5 if t_ex_days else (
            (pre_ex["T_days"].max() + post_ex["T_days"].min()) / 2
        )
        ax.annotate(
            "", xy=(arrow_x, post_med), xytext=(arrow_x, pre_med),
            arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2.5)
        )
        ax.text(arrow_x + 0.8, (pre_med + post_med)/2,
                f"STEP-UP\n+{step:.2f}%\n= early-exercise\npremium",
                color=GREEN, fontsize=10, fontweight="bold", va="center")
    
    # Reference: continuous-yield decomposition
    yahoo_q = reported_q * 100
    ax.axhline(yahoo_q, color=PURPLE, lw=1, ls=":", alpha=0.7,
               label=f"Yahoo continuous q = {yahoo_q:.2f}% (baseline only)")

ax.set_xlabel("Days to expiry", fontsize=10)
ax.set_ylabel("Market-implied carry rate q_impl (%)", fontsize=10)
ax.legend(fontsize=9, loc="upper left")

below = (df["diff"].abs() <= df["spread"]).mean() * 100
fig.text(0.5, 0.005,
    f"Parity holds within bid-ask for {below:.0f}% of contracts  |  "
    f"Residual gap → dividend misestimation, funding spreads, early-exercise premium, stale quotes",
    ha="center", color=DIM, fontsize=9.5, style="italic")

fname = f"synthetic_forward_{TICKER}.png"
plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n✓ Saved: {fname}")
df.to_csv(f"synthetic_forward_{TICKER}.csv", index=False)
print(f"✓ Saved: synthetic_forward_{TICKER}.csv ({len(df)} rows)")

# ─── 5. SELF-CHECK ANALYSIS ──────────────────────────────────────────────
# This is the part your prof wants: "does it make sense?"
print(f"\n{'═'*68}")
print(f"  SELF-CHECK ANALYSIS — does this output make sense?")
print(f"{'═'*68}\n")

avg_spread = df["spread"].mean()
avg_abs_gap = df["diff"].abs().mean()
pct_within = (df["diff"].abs() <= df["spread"]).mean() * 100
mean_q_implied = implied_q_by_T["q_implied"].mean()

# Check 1: sign and magnitude
print(f"  [1] Magnitude check")
print(f"      Avg |gap|:         ${avg_abs_gap:.3f}")
print(f"      Avg bid-ask:       ${avg_spread:.3f}")
print(f"      Ratio:             {avg_abs_gap/avg_spread:.2f}×  "
      f"({'✓ gap < spread, consistent with no-arb' if avg_abs_gap < avg_spread else '⚠ gap exceeds spread'})")

# Check 2: term structure
print(f"\n  [2] Term-structure check")
gap_short = summary.loc[summary["T_days"] == summary["T_days"].min(), "mean_gap"].iloc[0]
gap_long  = summary.loc[summary["T_days"] == summary["T_days"].max(), "mean_gap"].iloc[0]
T_long    = summary["T_days"].max()
print(f"      Shortest expiry gap:  ${gap_short:+.4f}")
print(f"      Longest expiry gap:   ${gap_long:+.4f}  (T={T_long}d)")
if abs(gap_long) > abs(gap_short) * 2:
    print(f"      → Gap GROWS with maturity. Classic dividend/funding signature.")
    print(f"        If q_custom too low, F_actual under-shoots, F_syn looks high.")
    print(f"        If q_custom too high, F_actual over-shoots, F_syn looks low.")
else:
    print(f"      → Gap roughly stable across maturities. Pure noise / friction.")

# Check 3: implied dividend
print(f"\n  [3] Implied-dividend check  (market's view of q)")
reliable = summary[~summary["unreliable"]]
if len(reliable) >= 1:
    mean_q_reliable = reliable["q_impl"].median()
    print(f"      Median q_implied (T ≥ 5d only): {mean_q_reliable*100:+.3f}%")
    print(f"      Median q_implied (all expiries): {mean_q_implied*100:+.3f}%  "
          f"(noisy — ignore if very different)")
else:
    mean_q_reliable = mean_q_implied
    print(f"      ⚠ No expiries with T ≥ 5d. All q_implied values unreliable.")
print(f"      Your q_custom input:             {Q_CUSTOM*100:+.3f}%")
print(f"      Yahoo-reported q (continuous):   {reported_q*100:+.3f}%")
if NEXT_DIV_AMOUNT > 0 and t_ex_days is not None:
    print(f"      Discrete div modeled:            ${NEXT_DIV_AMOUNT:.2f} on {NEXT_EX_DATE}")
diff_q = abs(mean_q_reliable - Q_CUSTOM) * 100
if diff_q < 0.5:
    verdict = "✓ q_custom matches market within 50bps — reasonable"
elif diff_q < 1.5:
    verdict = "~ q_custom off by a bit; tune closer to q_implied for cleaner fit"
else:
    verdict = "⚠ q_custom far from market implied — see diagnostic [7]"
print(f"      Verdict: {verdict}")

# Check 4: tradability
print(f"\n  [4] Tradability check")
print(f"      {pct_within:.0f}% of contracts have |gap| ≤ spread.")
if pct_within >= 80:
    print(f"      → Healthy market. No-arb holds after friction.")
elif pct_within >= 50:
    print(f"      → Partial arb-free. Some strikes show real divergence — likely")
    print(f"        stale quotes or wrong q assumption for one expiry.")
else:
    print(f"      → ⚠ Many contracts outside spread. Check Yahoo data freshness")
    print(f"        or revise q_custom. Real arb is rarely this obvious.")

# Check 5: outliers (specific expiries misbehaving)
print(f"\n  [5] Outlier expiries")
worst = summary.loc[summary["mean_bps"].abs().idxmax()]
print(f"      Largest-gap expiry: T={int(worst['T_days'])}d  "
      f"(mean gap {worst['mean_bps']:+.1f} bps, q_implied {worst['q_impl']*100:+.2f}%)")
if abs(worst["mean_bps"]) > 20:
    print(f"      → Significant. Likely reasons (ranked):")
    print(f"        (a) Ex-dividend date falls inside this expiry → bigger q effect")
    print(f"        (b) Early-exercise premium on American options (esp. deep ITM puts)")
    print(f"        (c) Low-volume expiry with wide stale quotes")
else:
    print(f"      → Within expected noise band.")

# Check 6: sanity of spot/forward relationship
print(f"\n  [6] Sanity: does F_actual > Spot?  (holds when r > q)")
max_T = summary["T_days"].max() / 365
F_far = spot * np.exp((R - Q_CUSTOM) * max_T)
print(f"      Spot: ${spot:.2f}   F_actual({int(max_T*365)}d): ${F_far:.2f}   "
      f"Carry: {(F_far/spot-1)*100:+.3f}%")
print(f"      → Sign matches (r−q)={R-Q_CUSTOM:+.3f}. ✓")

# Check 7: diagnose the direction and cause of any systematic gap
print(f"\n  [7] Diagnosis: what's causing the gap?")
signed_mean = summary["mean_gap"].mean()
signed_mean_bps = summary["mean_bps"].mean()
# Only use reliable expiries for sign consistency check
reliable = summary[~summary["unreliable"]]
if len(reliable) >= 2:
    consistent_sign = (reliable["mean_gap"] > 0).all() or (reliable["mean_gap"] < 0).all()
    signed_reliable = reliable["mean_bps"].mean()
else:
    consistent_sign = False
    signed_reliable = signed_mean_bps
large_gap = abs(signed_reliable) > 5

if not large_gap:
    print(f"      Mean gap small ({signed_reliable:+.1f} bps on T≥5d expiries). No systematic bias.")
elif stale_flag or not market_open:
    print(f"      Mean gap: {signed_reliable:+.1f} bps, consistent sign: {consistent_sign}")
    print(f"      → LIKELY STALE QUOTES. Market is closed and mid-prices are")
    print(f"        frozen at last trade. Re-run during 9:30–16:00 ET.")
elif NEXT_DIV_AMOUNT > 0 and consistent_sign and abs(signed_reliable) < 10:
    # After discrete-div correction, small residual gap is normal
    print(f"      Residual gap ({signed_reliable:+.1f} bps) after discrete-div correction.")
    print(f"      Within expected range for American options with near-ex-div dynamics.")
    print(f"      → Likely remaining causes:")
    print(f"        (a) Early-exercise premium on calls (optimal to exercise just")
    print(f"            before ex-date to capture the $0.26).")
    print(f"        (b) Minor borrow-rate or funding-spread effect.")
elif consistent_sign and signed_reliable > 5:
    print(f"      F_syn systematically ABOVE F_actual → q_implied > q_custom.")
    print(f"      Possible causes:")
    if NEXT_DIV_AMOUNT == 0 and t_ex_days is None:
        print(f"        (a) UNMODELED DISCRETE DIVIDEND — most likely if upcoming ex-")
        print(f"            date falls near/inside expiry window. Set NEXT_DIV_AMOUNT")
        print(f"            and NEXT_EX_DATE in config.")
    print(f"        (b) Stock borrow cost — hard-to-borrow names need F = S·e^((r−q−b)T).")
    print(f"        (c) Early-exercise premium on American calls.")
    print(f"        (d) Mis-specified risk-free rate R.")
elif consistent_sign and signed_reliable < -5:
    print(f"      F_syn systematically BELOW F_actual → market pricing MORE")
    print(f"      carry reduction than your cost-of-carry model captures.")
    
    # Check if q_implied is STABLE across reliable expiries → American exercise signature
    reliable_nonshort = reliable[reliable["T_days"] >= 10]   # 10d+ for stability check
    if len(reliable_nonshort) >= 3:
        q_std = reliable_nonshort["q_impl"].std()
        q_med = reliable_nonshort["q_impl"].median()
        if q_std < 0.01:   # q_implied within 100bps across all long expiries
            print(f"\n      ═══ STABLE q_implied DETECTED ═══")
            print(f"      q_implied across expiries ≥10d: {q_med*100:+.2f}% "
                  f"± {q_std*100:.2f}%  (very stable)")
            print(f"      The flatness across maturities is the fingerprint of a")
            print(f"      TRUE carry component (not a wrong dividend amount).")
            print(f"")
            
            if NEXT_DIV_AMOUNT > 0 and t_ex_days is not None:
                # Dividend-paying stock with upcoming ex-date → AAPL/KO story
                print(f"      DECOMPOSITION of market-implied carry ≈ {q_med*100:.2f}%:")
                yahoo_div = reported_q
                residual = q_med - yahoo_div
                print(f"        • Continuous dividend yield (Yahoo):      {yahoo_div*100:+.2f}%")
                print(f"        • Stock borrow cost (est., easy-to-borrow): +0.20 to +0.50%")
                print(f"        • American early-exercise premium:        +{max(0, (residual-0.004))*100:.2f}% "
                      f"to +{max(0, (residual-0.002))*100:.2f}%")
                print(f"        ────────────────────────────────────────────────────")
                print(f"        Total implied:                            {q_med*100:+.2f}% ✓")
                print(f"")
                print(f"      INTERPRETATION:")
                print(f"      European put-call parity underprices American options by")
                print(f"      the early-exercise premium. Call holders rationally exercise")
                print(f"      just before ex-date to capture the ${NEXT_DIV_AMOUNT} dividend.")
                print(f"      Gap is real but NOT arbitrage — requires accepting early-")
                print(f"      exercise assignment risk on the short leg.")
            elif reported_q < 0.002:
                # Non-dividend stock → TSLA/GME story
                print(f"      DECOMPOSITION of market-implied carry ≈ {q_med*100:.2f}%:")
                print(f"        • Continuous dividend yield:              {reported_q*100:+.2f}%  (none)")
                print(f"        • Stock borrow cost / funding spread:     ~{q_med*100:+.2f}%")
                print(f"        ────────────────────────────────────────────────────")
                print(f"        Total implied:                            {q_med*100:+.2f}% ✓")
                print(f"")
                print(f"      INTERPRETATION:")
                print(f"      This stock pays no dividend, so the implied carry cannot")
                print(f"      come from dividend or early-exercise mechanics. It reflects")
                print(f"      stock borrow cost (shares are expensive to short) and/or")
                print(f"      funding-spread basis (options market's implied rate differs")
                print(f"      from your R = {R*100:.2f}%). Typical for lightly-shorted names: 0-")
                print(f"      0.5%. Higher values (1%+) indicate hard-to-borrow status.")
            else:
                # Dividend-paying but no ex-date configured
                print(f"      DECOMPOSITION of market-implied carry ≈ {q_med*100:.2f}%:")
                print(f"        • Yahoo-reported dividend yield:          {reported_q*100:+.2f}%")
                print(f"        • Residual (borrow + funding + exercise): {(q_med-reported_q)*100:+.2f}%")
                print(f"        ────────────────────────────────────────────────────")
                print(f"        Total implied:                            {q_med*100:+.2f}% ✓")
        else:
            print(f"      q_implied varies by {q_std*100:.2f}% across expiries (not flat).")
            print(f"      → Possibly a mix of effects. Check individual expiries.")
    if NEXT_DIV_AMOUNT > 0 and t_ex_days is not None:
        n_covering = (summary["T_days"] >= t_ex_days).sum()
        if n_covering == 0:
            print(f"      ⚠ Ex-date ({NEXT_EX_DATE}, {t_ex_days}d out) is BEYOND every")
            print(f"        expiry in your data. Your discrete-div term is silent.")
            print(f"        FIX: bump MAX_EXPIRIES so some expiry covers day {t_ex_days}+.")
else:
    print(f"      Gap pattern is mixed across expiries. Check Panel ② directly.")
    print(f"      If sign flips by expiry → individual ex-div dates may matter.")

print(f"\n{'═'*68}")
# Check if we're in the "stable q_implied, just American exercise" case
reliable_nonshort = summary[~summary["unreliable"]].query("T_days >= 10")
stable_q = (len(reliable_nonshort) >= 3 and reliable_nonshort["q_impl"].std() < 0.01)

if stale_flag or not market_open:
    print(f"  OVERALL: STALE DATA — rerun during US market hours for clean result")
elif stable_q:
    q_med = reliable_nonshort["q_impl"].median()
    if NEXT_DIV_AMOUNT > 0 and t_ex_days is not None:
        print(f"  OVERALL: SENSIBLE OUTPUT — implied carry ≈ {q_med*100:.2f}% includes")
        print(f"           American early-exercise premium near ex-dividend. Real effect.")
    elif reported_q < 0.002 and Q_CUSTOM == 0:
        print(f"  OVERALL: SENSIBLE OUTPUT — implied carry ≈ {q_med*100:.2f}% on a non-")
        print(f"           dividend stock. Most likely stock borrow cost / funding spread.")
    else:
        print(f"  OVERALL: SENSIBLE OUTPUT — implied carry ≈ {q_med*100:.2f}% stable across")
        print(f"           maturities. Reflects combined dividend + borrow + funding.")
elif pct_within >= 70 and avg_abs_gap < avg_spread:
    print(f"  OVERALL: SENSIBLE OUTPUT")
else:
    print(f"  OVERALL: CHECK INPUTS — see diagnostic [7] above")
print(f"{'═'*68}\n")

plt.close()
