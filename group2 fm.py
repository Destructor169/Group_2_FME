"""
═══════════════════════════════════════════════════════════════════════
  SYNTHETIC FORWARD ANALYSIS — with dividend yield + interpretability
═══════════════════════════════════════════════════════════════════════

  Put-call parity:  F_syn(K) = K + (C - P)·e^(rT)
  Cost-of-carry:    F_actual = (S - PV_div)·e^((r-q)T)

  v3: Handles Yahoo returning zero/NaN OI and bid/ask outside market hours
  by falling back to lastPrice and auto-relaxing the OI filter.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime, timezone, timedelta as td
import yfinance as yf

# ─── CONFIG ──────────────────────────────────────────────────────────────
TICKER        = "AAPL"
R             = 0.045

Q_CUSTOM           = 0.0
NEXT_DIV_AMOUNT    = 0.26
NEXT_EX_DATE       = "2026-05-12"

MIN_OI        = 20
ATM_WINDOW    = 0.05
MAX_EXPIRIES  = 10
RELAX_OI_FILTER = True     # Auto-skip OI filter when Yahoo returns 0/NaN OI

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
    if reported_q > 0.15: reported_q /= 100
except Exception:
    reported_q = 0.0

spot = yahoo_spot

print(f"  Spot (Yahoo, initial): ${yahoo_spot:.2f}")
print(f"  Risk-free rate r:      {R*100:.2f}%")
print(f"  Reported div yield:    {reported_q*100:.2f}%  (from Yahoo)")
print(f"  USER-SET q:            {Q_CUSTOM*100:.2f}%  ← used in F_actual")
print(f"  Filters:               OI ≥ {MIN_OI} (auto-relax={RELAX_OI_FILTER}), strikes within ±{ATM_WINDOW*100:.0f}% of spot")
print(f"  Max expiries scanned:  {MAX_EXPIRIES}\n")

print(f"  {'─'*60}")
print(f"  Loading option chains...")
print(f"  {'─'*60}")

rows = []
oi_relaxed_any = False
price_fallback_any = False

for exp_str in tk.options[:MAX_EXPIRIES]:
    exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
    T_days = (exp - today).days
    T = T_days / 365
    if T <= 0: continue

    oc = tk.option_chain(exp_str)
    call_cols = ["bid","ask","openInterest","volume","lastPrice","lastTradeDate"]
    c = oc.calls.set_index("strike")[[col for col in call_cols if col in oc.calls.columns]]
    p = oc.puts .set_index("strike")[[col for col in call_cols if col in oc.puts.columns]]
    c.columns = [f"c_{col.replace('openInterest','oi').replace('lastPrice','last').replace('lastTradeDate','ltd')}" for col in c.columns]
    p.columns = [f"p_{col.replace('openInterest','oi').replace('lastPrice','last').replace('lastTradeDate','ltd')}" for col in p.columns]
    chain = c.join(p, how="inner").reset_index().rename(columns={"strike":"K"})

    before = len(chain)

    # ATM window first
    chain = chain[chain["K"].between(spot*(1-ATM_WINDOW), spot*(1+ATM_WINDOW))]
    after_atm = len(chain)

    # Fill bid/ask NaNs with 0 so comparisons work
    for col in ["c_bid","c_ask","p_bid","p_ask"]:
        if col in chain.columns:
            chain[col] = chain[col].fillna(0)

    # Fallback prices: use lastPrice when bid/ask are zero
    c_last = chain["c_last"] if "c_last" in chain.columns else 0
    p_last = chain["p_last"] if "p_last" in chain.columns else 0
    chain["c_bid_eff"] = chain["c_bid"].where(chain["c_bid"] > 0, c_last)
    chain["c_ask_eff"] = chain["c_ask"].where(chain["c_ask"] > 0, c_last)
    chain["p_bid_eff"] = chain["p_bid"].where(chain["p_bid"] > 0, p_last)
    chain["p_ask_eff"] = chain["p_ask"].where(chain["p_ask"] > 0, p_last)

    chain["used_price_fallback"] = (
        (chain["c_bid"] == 0) | (chain["c_ask"] == 0) |
        (chain["p_bid"] == 0) | (chain["p_ask"] == 0)
    )

    # Require valid effective prices
    chain = chain[(chain["c_bid_eff"]>0) & (chain["c_ask_eff"]>0)
                 &(chain["p_bid_eff"]>0) & (chain["p_ask_eff"]>0)]

    # OI filter with auto-relax
    oi_c_ok = (chain["c_oi"].fillna(0) >= MIN_OI).sum() if "c_oi" in chain.columns else 0
    oi_p_ok = (chain["p_oi"].fillna(0) >= MIN_OI).sum() if "p_oi" in chain.columns else 0
    oi_relaxed_this = False

    if RELAX_OI_FILTER and (oi_c_ok == 0 or oi_p_ok == 0):
        oi_relaxed_this = True
        oi_relaxed_any = True
    else:
        if "c_oi" in chain.columns and "p_oi" in chain.columns:
            chain = chain[(chain["c_oi"].fillna(0)>=MIN_OI) & (chain["p_oi"].fillna(0)>=MIN_OI)]

    chain["C"] = (chain["c_bid_eff"]+chain["c_ask_eff"])/2
    chain["P"] = (chain["p_bid_eff"]+chain["p_ask_eff"])/2

    raw_spread = 0.5*((chain["c_ask_eff"]-chain["c_bid_eff"]) + (chain["p_ask_eff"]-chain["p_bid_eff"]))
    proxy_spread = 0.005 * (chain["C"] + chain["P"]) / 2
    chain["spread"] = raw_spread.where(raw_spread > 0, proxy_spread)

    chain["T_days"] = T_days
    chain["T"] = T
    chain["expiry"] = exp

    if chain["used_price_fallback"].any():
        price_fallback_any = True

    n_fb = chain["used_price_fallback"].sum()
    flags = []
    if n_fb > 0: flags.append(f"price-fb:{n_fb}")
    if oi_relaxed_this: flags.append("oi-skip")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    print(f"    {exp} (T={T_days:>3}d): {before:>3} strikes → {after_atm:>2} ATM → {len(chain):>2} usable{flag_str}")
    if len(chain): rows.append(chain)

if not rows:
    print("\n  ⚠ No data even with fallbacks. Try different ticker.")
    raise SystemExit

df = pd.concat(rows, ignore_index=True)

print()
if price_fallback_any:
    total_fb = df["used_price_fallback"].sum()
    pct_fb = 100 * total_fb / len(df)
    print(f"  ⚠ {total_fb}/{len(df)} rows ({pct_fb:.0f}%) used lastPrice fallback")
    print(f"    Results are INDICATIVE (market likely closed).")
if oi_relaxed_any:
    print(f"  ⚠ OI filter auto-relaxed (Yahoo returned zero/NaN OI —")
    print(f"    typical outside US market hours).")

# Market hours
now_utc = datetime.now(timezone.utc)
now_et  = now_utc - td(hours=4)
et_hour = now_et.hour + now_et.minute/60
is_weekday = now_et.weekday() < 5
market_open = is_weekday and (9.5 <= et_hour <= 16.0)

print(f"\n  {'─'*60}")
print(f"  Quote-freshness check")
print(f"  {'─'*60}")
print(f"    Current time (ET approx):  {now_et.strftime('%Y-%m-%d %H:%M')}  "
      f"({'weekday' if is_weekday else 'weekend'})")
print(f"    US options market open:    {'✓ YES' if market_open else '✗ NO — quotes likely stale'}")

stale_flag = False

# Option-implied spot
atm_probe = df.iloc[(df["K"] - yahoo_spot).abs().argsort()[:5]].copy()
atm_probe["S_impl"] = (atm_probe["C"] - atm_probe["P"]
                      + atm_probe["K"] * np.exp(-R * atm_probe["T"]))
implied_spot = float(atm_probe["S_impl"].median())
spot_drift = implied_spot - yahoo_spot

print(f"\n  {'─'*60}")
print(f"  Spot consistency")
print(f"  {'─'*60}")
print(f"    Yahoo-reported spot:       ${yahoo_spot:.2f}")
print(f"    Spot implied by options:   ${implied_spot:.2f}  (Δ = {spot_drift:+.2f})")

spot = implied_spot
if abs(spot_drift) > 0.15:
    print(f"    ⚠ ${abs(spot_drift):.2f} drift. Using option-implied ${spot:.2f}.")
else:
    print(f"    ✓ Spot sources agree.")

if "c_ltd" in df.columns:
    try:
        df["c_ltd"] = pd.to_datetime(df["c_ltd"], utc=True, errors="coerce")
        df["p_ltd"] = pd.to_datetime(df["p_ltd"], utc=True, errors="coerce")
        latest_quote = max(df["c_ltd"].max(), df["p_ltd"].max())
        age_hours = (now_utc - latest_quote).total_seconds() / 3600
        print(f"    Most recent quote age:     {age_hours:.1f} hours old")
        if age_hours > 2 and not market_open:
            stale_flag = True
            print(f"    ⚠ Quotes frozen from last session. INDICATIVE only.")
    except Exception as e:
        print(f"    (couldn't parse lastTradeDate: {e})")

# ─── 2. CONSTRUCT SYNTHETIC FORWARD ──────────────────────────────────────
ex_date = None
t_ex_days = None
if NEXT_DIV_AMOUNT > 0 and NEXT_EX_DATE:
    ex_date = datetime.strptime(NEXT_EX_DATE, "%Y-%m-%d").date()
    t_ex_days = (ex_date - today).days
    print(f"\n  Discrete div:          ${NEXT_DIV_AMOUNT:.2f} on {NEXT_EX_DATE} "
          f"(t_ex = {t_ex_days}d)")

df["F_syn"] = df["K"] + (df["C"] - df["P"]) * np.exp(R * df["T"])

def pv_div(T_days_row):
    if t_ex_days is None or t_ex_days > T_days_row or t_ex_days < 0:
        return 0.0
    t_ex = t_ex_days / 365
    return NEXT_DIV_AMOUNT * np.exp(-R * t_ex)

df["PV_div"]   = df["T_days"].apply(pv_div)
df["F_actual"] = (spot - df["PV_div"]) * np.exp((R - Q_CUSTOM) * df["T"])
df["F_actual_noq"] = spot * np.exp(R * df["T"])

if t_ex_days is not None:
    n_covering = (df["T_days"] >= t_ex_days).sum()
    if n_covering == 0:
        max_T = df["T_days"].max()
        print(f"\n  ⚠ ex-date is {t_ex_days}d out, longest expiry {max_T}d. "
              f"Discrete-div has no effect.")

df["diff"]     = df["F_syn"] - df["F_actual"]
df["diff_bps"] = 1e4 * df["diff"] / df["F_actual"]

implied_q_by_T = (
    df.groupby("T_days")[["F_syn","T"]]
      .apply(lambda g: R - np.log(g["F_syn"].median() / spot) / g["T"].iloc[0], include_groups=False)
      .rename("q_implied")
      .reset_index()
)

# ─── 3. SUMMARY TABLE ────────────────────────────────────────────────────
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
    unreliable = Td < 5
    q_str = f"{q_impl*100:+.2f}%" + (" *" if unreliable else "  ")
    summary_rows.append({"T_days":Td,"mean_gap":mean_gap,"std":std_gap,
                         "mean_bps":mean_bps,"frac_tradable":frac_tradable,
                         "q_impl":q_impl,"n":len(g),"includes_ex":includes_ex,
                         "unreliable":unreliable})
    ex_mark = "✓" if includes_ex else "—"
    print(f"  {Td:>4}  {len(g):>3}  {ex_mark:>7}  {mean_gap:>+11.4f}  "
          f"{mean_bps:>+9.2f}  {frac_tradable:>9.0f}%  {q_str:>8}")
print(f"\n  * = T < 5d, q_implied unreliable")
if t_ex_days is not None:
    print(f"  ✓ = ex-dividend date (day {t_ex_days}) falls INSIDE this expiry")
summary = pd.DataFrame(summary_rows)

# ─── 4. PLOTS ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 13.5))
gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.28, height_ratios=[1, 1, 1.15])
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

best_T = df.groupby("T_days").size().idxmax()
sub = df[df["T_days"] == best_T].sort_values("K")

ax = fig.add_subplot(gs[0, 0])
clean(ax, f"① Construction check  —  T = {best_T} days",
      "Gold dots = synthetic forward per strike. Blue line = theoretical forward.")
F_act_val = sub["F_actual"].iloc[0]
F_act_noq = sub["F_actual_noq"].iloc[0]
pv_d = sub["PV_div"].iloc[0]
blue_label = (f"F_actual = (S−PV_div)·e^(rT) = \\${F_act_val:.3f}" if pv_d > 0
              else f"F_actual = S·e^((r−q)T) = \\${F_act_val:.3f}")
ax.axhline(F_act_val, color=BLUE, lw=2, zorder=2, label=blue_label)
ax.axhline(F_act_noq, color=RED, lw=1.2, ls=":", alpha=0.7, zorder=1,
           label=f"F_actual (no div, no q) = \\${F_act_noq:.3f}")
ax.scatter(sub["K"], sub["F_syn"], s=60, color=GOLD,
           edgecolor=BG, lw=0.6, zorder=3, label="F_syn = K + (C−P)·e^(rT)")
ymin = min(sub["F_syn"].min(), F_act_val, F_act_noq)
ymax = max(sub["F_syn"].max(), F_act_val, F_act_noq)
pad = (ymax - ymin) * 0.3 + 0.1
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_xlabel("Strike K (\\$)"); ax.set_ylabel("Forward price (\\$)")
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

ax = fig.add_subplot(gs[0, 1])
clean(ax, "② Gap across strikes, by expiry",
      "Flat lines at 0 = parity holds. Drift with K = friction/skew.")
cmap = plt.get_cmap("plasma")
Ts = sorted(df["T_days"].unique())
for Td, col in zip(Ts, cmap(np.linspace(0.15, 0.85, len(Ts)))):
    g = df[df["T_days"] == Td].sort_values("K")
    ax.plot(g["K"], g["diff_bps"], "o-", color=col, lw=1.5, ms=5,
            alpha=0.9, label=f"{Td}d")
ax.axhline(0, color=FG, lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Strike K (\\$)"); ax.set_ylabel("Gap (bps of F)")
ax.legend(fontsize=8, ncol=2, title="Maturity", title_fontsize=8)

ax = fig.add_subplot(gs[1, 0])
clean(ax, "③ Mean gap vs time to expiry",
      "Gold band = bid-ask friction. Blue inside band → no arb.")
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

ax = fig.add_subplot(gs[1, 1])
clean(ax, "④ Is the gap tradable?  |gap|  vs  bid-ask spread",
      "Above dashed = noise. Below = potential arb.")
sc = ax.scatter(df["diff"].abs(), df["spread"], c=df["T_days"],
                cmap="plasma", s=32, alpha=0.8, edgecolor=BG, lw=0.3)
mx = max(df["diff"].abs().quantile(0.98), df["spread"].quantile(0.98)) * 1.15
ax.plot([0, mx], [0, mx], ls="--", color=FG, lw=1, alpha=0.6, label="|gap| = spread")
xx = np.linspace(0, mx, 100)
ax.fill_between(xx, xx, mx, color=GREEN, alpha=0.10,
                label="Explained by friction")
ax.fill_between(xx, 0, xx, color=RED, alpha=0.10,
                label="Unexplained")
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.set_xlabel("|Gap|  (\\$)"); ax.set_ylabel("Bid-ask spread  (\\$)")
cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Days to expiry", color=DIM, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=DIM)
ax.legend(fontsize=8, loc="lower right")

ax = fig.add_subplot(gs[2, :])
clean(ax, "⑤ The ex-dividend step  —  q_implied by expiry",
      "Step across ex-date = American early-exercise premium.")

plot_df = summary[~summary["unreliable"]].copy().sort_values("T_days")
if len(plot_df) > 0:
    pre_ex  = plot_df[~plot_df["includes_ex"]]
    post_ex = plot_df[ plot_df["includes_ex"]]

    ax.plot(plot_df["T_days"], plot_df["q_impl"]*100,
            color=DIM, lw=1, ls=":", alpha=0.5, zorder=1)

    if len(pre_ex):
        ax.scatter(pre_ex["T_days"], pre_ex["q_impl"]*100,
                   s=140, color=BLUE, edgecolor=FG, lw=1.2, zorder=3,
                   label="Ex-date OUTSIDE window")
        for _, row in pre_ex.iterrows():
            ax.annotate(f"{row['q_impl']*100:+.2f}%",
                        (row["T_days"], row["q_impl"]*100),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", color=BLUE, fontsize=9, fontweight="bold")

    if len(post_ex):
        ax.scatter(post_ex["T_days"], post_ex["q_impl"]*100,
                   s=140, color=GOLD, edgecolor=FG, lw=1.2, zorder=3,
                   marker="D", label="Ex-date INSIDE window")
        for _, row in post_ex.iterrows():
            ax.annotate(f"{row['q_impl']*100:+.2f}%",
                        (row["T_days"], row["q_impl"]*100),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", color=GOLD, fontsize=9, fontweight="bold")

    if t_ex_days is not None:
        ax.axvline(t_ex_days, color=RED, lw=2, ls="--", alpha=0.7, zorder=2)
        ylim_now = ax.get_ylim()
        ax.text(t_ex_days, ylim_now[0] + (ylim_now[1]-ylim_now[0])*0.05,
                f" ex-date (day {t_ex_days})",
                color=RED, fontsize=10, fontweight="bold", va="bottom", ha="left")

    if len(pre_ex) and len(post_ex):
        pre_med  = pre_ex["q_impl"].median() * 100
        post_med = post_ex["q_impl"].median() * 100
        step     = post_med - pre_med
        ax.axhline(pre_med,  color=BLUE, lw=1, ls="-", alpha=0.4, zorder=1)
        ax.axhline(post_med, color=GOLD, lw=1, ls="-", alpha=0.4, zorder=1)
        arrow_x = t_ex_days + 1.5 if t_ex_days else (
            (pre_ex["T_days"].max() + post_ex["T_days"].min()) / 2
        )
        ax.annotate("", xy=(arrow_x, post_med), xytext=(arrow_x, pre_med),
                    arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2.5))
        ax.text(arrow_x + 0.8, (pre_med + post_med)/2,
                f"STEP-UP\n+{step:.2f}%",
                color=GREEN, fontsize=10, fontweight="bold", va="center")

    yahoo_q = reported_q * 100
    ax.axhline(yahoo_q, color=PURPLE, lw=1, ls=":", alpha=0.7,
               label=f"Yahoo continuous q = {yahoo_q:.2f}%")

ax.set_xlabel("Days to expiry", fontsize=10)
ax.set_ylabel("Market-implied carry rate q_impl (%)", fontsize=10)
ax.legend(fontsize=9, loc="upper left")

below = (df["diff"].abs() <= df["spread"]).mean() * 100
fig.text(0.5, 0.005,
    f"Parity holds within bid-ask for {below:.0f}% of contracts  |  "
    f"Residual → dividend/funding/exercise premium/stale quotes",
    ha="center", color=DIM, fontsize=9.5, style="italic")

fname = f"synthetic_forward_{TICKER}.png"
plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n✓ Saved: {fname}")
df.to_csv(f"synthetic_forward_{TICKER}.csv", index=False)
print(f"✓ Saved: synthetic_forward_{TICKER}.csv ({len(df)} rows)")

# ─── 5. SELF-CHECK ────────────────────────────────────────────────────────
print(f"\n{'═'*68}")
print(f"  SELF-CHECK ANALYSIS")
print(f"{'═'*68}\n")

avg_spread = df["spread"].mean()
avg_abs_gap = df["diff"].abs().mean()
pct_within = (df["diff"].abs() <= df["spread"]).mean() * 100
mean_q_implied = implied_q_by_T["q_implied"].mean()

print(f"  [1] Magnitude check")
print(f"      Avg |gap|:         ${avg_abs_gap:.3f}")
print(f"      Avg bid-ask:       ${avg_spread:.3f}")
print(f"      Ratio:             {avg_abs_gap/avg_spread:.2f}×  "
      f"({'✓ no-arb' if avg_abs_gap < avg_spread else '⚠ gap > spread'})")

print(f"\n  [2] Term-structure check")
gap_short = summary.loc[summary["T_days"] == summary["T_days"].min(), "mean_gap"].iloc[0]
gap_long  = summary.loc[summary["T_days"] == summary["T_days"].max(), "mean_gap"].iloc[0]
print(f"      Shortest expiry gap:  ${gap_short:+.4f}")
print(f"      Longest expiry gap:   ${gap_long:+.4f}")

print(f"\n  [3] Implied-dividend check")
reliable = summary[~summary["unreliable"]]
if len(reliable) >= 1:
    mean_q_reliable = reliable["q_impl"].median()
    print(f"      Median q_implied (T ≥ 5d): {mean_q_reliable*100:+.3f}%")
else:
    mean_q_reliable = mean_q_implied
    print(f"      ⚠ No reliable expiries.")
print(f"      Your q_custom:             {Q_CUSTOM*100:+.3f}%")
print(f"      Yahoo-reported q:          {reported_q*100:+.3f}%")

print(f"\n  [4] Tradability: {pct_within:.0f}% of contracts have |gap| ≤ spread")

print(f"\n  [5] Outliers")
worst = summary.loc[summary["mean_bps"].abs().idxmax()]
print(f"      Largest gap: T={int(worst['T_days'])}d ({worst['mean_bps']:+.1f} bps, "
      f"q_impl {worst['q_impl']*100:+.2f}%)")

print(f"\n  [6] Spot/forward sanity")
max_T = summary["T_days"].max() / 365
F_far = spot * np.exp((R - Q_CUSTOM) * max_T)
print(f"      Spot ${spot:.2f} → F({int(max_T*365)}d) ${F_far:.2f}  "
      f"({(F_far/spot-1)*100:+.3f}%) ✓")

print(f"\n  [7] Diagnosis")
signed_reliable = reliable["mean_bps"].mean() if len(reliable) >= 2 else summary["mean_bps"].mean()

if stale_flag or not market_open:
    print(f"      Stale quotes likely. Rerun during US market hours.")
elif abs(signed_reliable) <= 5:
    print(f"      Mean gap small ({signed_reliable:+.1f} bps). Clean.")
elif signed_reliable > 5:
    print(f"      F_syn > F_actual by {signed_reliable:.1f} bps → q_implied > q_custom.")
    print(f"      Likely unmodeled dividend, borrow cost, or early-exercise premium.")
else:
    print(f"      F_syn < F_actual by {abs(signed_reliable):.1f} bps → q_implied < q_custom.")
    print(f"      Likely overestimated q_custom, or borrow-related carry.")

print(f"\n{'═'*68}")
if stale_flag or not market_open:
    print(f"  OVERALL: STALE DATA — rerun during US market hours for clean result")
elif pct_within >= 70 and avg_abs_gap < avg_spread:
    print(f"  OVERALL: SENSIBLE OUTPUT")
else:
    print(f"  OVERALL: CHECK INPUTS — see [7]")
print(f"{'═'*68}\n")

plt.show()
