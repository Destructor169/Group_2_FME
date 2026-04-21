import io
from datetime import date, datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=600)
def run_analysis(
    ticker: str,
    r_rate: float,
    q_custom: float,
    next_div_amount: float,
    next_ex_date: str | None,
    min_oi: int,
    atm_window: float,
    max_expiries: int,
) -> dict:
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        raise ValueError("No price history returned for this ticker.")

    yahoo_spot = float(hist["Close"].iloc[-1])
    today = date.today()

    try:
        info = tk.info
        reported_q = info.get("dividendYield", 0) or 0
        if reported_q > 0.15:
            reported_q /= 100
    except Exception:
        reported_q = 0.0

    rows = []
    chain_counts = []
    spot_ref = yahoo_spot

    options = tk.options[:max_expiries]
    if not options:
        raise ValueError("No options expiries found for this ticker.")

    for exp_str in options:
        exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
        t_days = (exp - today).days
        t_years = t_days / 365
        if t_years <= 0:
            continue

        oc = tk.option_chain(exp_str)
        call_cols = ["bid", "ask", "openInterest", "volume", "lastPrice", "lastTradeDate"]
        put_cols = call_cols

        c = oc.calls.set_index("strike")[[col for col in call_cols if col in oc.calls.columns]]
        p = oc.puts.set_index("strike")[[col for col in put_cols if col in oc.puts.columns]]
        c.columns = [
            f"c_{col.replace('openInterest', 'oi').replace('lastPrice', 'last').replace('lastTradeDate', 'ltd')}"
            for col in c.columns
        ]
        p.columns = [
            f"p_{col.replace('openInterest', 'oi').replace('lastPrice', 'last').replace('lastTradeDate', 'ltd')}"
            for col in p.columns
        ]

        chain = c.join(p, how="inner").reset_index().rename(columns={"strike": "K"})
        before = len(chain)

        chain = chain[
            (chain["c_bid"] > 0)
            & (chain["c_ask"] > 0)
            & (chain["p_bid"] > 0)
            & (chain["p_ask"] > 0)
            & (chain["c_oi"] >= min_oi)
            & (chain["p_oi"] >= min_oi)
        ]
        chain = chain[chain["K"].between(spot_ref * (1 - atm_window), spot_ref * (1 + atm_window))]

        if chain.empty:
            chain_counts.append({"expiry": exp_str, "t_days": t_days, "before": before, "after": 0})
            continue

        chain["C"] = (chain["c_bid"] + chain["c_ask"]) / 2
        chain["P"] = (chain["p_bid"] + chain["p_ask"]) / 2
        chain["spread"] = 0.5 * ((chain["c_ask"] - chain["c_bid"]) + (chain["p_ask"] - chain["p_bid"]))
        chain["T_days"] = t_days
        chain["T"] = t_years
        chain["expiry"] = exp

        rows.append(chain)
        chain_counts.append({"expiry": exp_str, "t_days": t_days, "before": before, "after": len(chain)})

    if not rows:
        raise ValueError("No liquid ATM options passed the filters. Try lower OI or wider ATM window.")

    df = pd.concat(rows, ignore_index=True)

    now_utc = datetime.now(timezone.utc)
    now_et = now_utc - timedelta(hours=4)
    et_hour = now_et.hour + now_et.minute / 60
    is_weekday = now_et.weekday() < 5
    market_open = is_weekday and (9.5 <= et_hour <= 16.0)

    # Option-implied spot keeps spot and options on the same quote timestamp.
    atm_probe = df.iloc[(df["K"] - yahoo_spot).abs().argsort()[:5]].copy()
    atm_probe["S_impl"] = atm_probe["C"] - atm_probe["P"] + atm_probe["K"] * np.exp(-r_rate * atm_probe["T"])
    implied_spot = float(atm_probe["S_impl"].median())
    spot = implied_spot
    spot_drift = spot - yahoo_spot

    stale_flag = False
    quote_age_hours = None
    if "c_ltd" in df.columns and "p_ltd" in df.columns:
        try:
            df["c_ltd"] = pd.to_datetime(df["c_ltd"], utc=True, errors="coerce")
            df["p_ltd"] = pd.to_datetime(df["p_ltd"], utc=True, errors="coerce")
            latest_quote = max(df["c_ltd"].max(), df["p_ltd"].max())
            if pd.notna(latest_quote):
                quote_age_hours = (now_utc - latest_quote).total_seconds() / 3600
                if quote_age_hours > 2 and not market_open:
                    stale_flag = True
        except Exception:
            pass

    ex_date = None
    t_ex_days = None
    if next_div_amount > 0 and next_ex_date:
        ex_date = datetime.strptime(next_ex_date, "%Y-%m-%d").date()
        t_ex_days = (ex_date - today).days

    df["F_syn"] = df["K"] + (df["C"] - df["P"]) * np.exp(r_rate * df["T"])

    def pv_div(t_days_row: int) -> float:
        if t_ex_days is None or t_ex_days > t_days_row or t_ex_days < 0:
            return 0.0
        t_ex = t_ex_days / 365
        return next_div_amount * np.exp(-r_rate * t_ex)

    df["PV_div"] = df["T_days"].apply(pv_div)
    df["F_actual"] = (spot - df["PV_div"]) * np.exp((r_rate - q_custom) * df["T"])
    df["F_actual_noq"] = spot * np.exp(r_rate * df["T"])
    df["diff"] = df["F_syn"] - df["F_actual"]
    df["diff_bps"] = 1e4 * df["diff"] / df["F_actual"]

    implied_q_by_t = (
        df.groupby("T_days")[["F_syn", "T"]]
        .apply(lambda g: r_rate - np.log(g["F_syn"].median() / spot) / g["T"].iloc[0], include_groups=False)
        .rename("q_implied")
        .reset_index()
    )

    summary_rows = []
    for t_days, g in df.groupby("T_days"):
        mean_gap = g["diff"].mean()
        std_gap = g["diff"].std() if len(g) > 1 else 0
        mean_bps = g["diff_bps"].mean()
        frac_tradable = (g["diff"].abs() > g["spread"]).mean() * 100
        q_impl = implied_q_by_t.loc[implied_q_by_t["T_days"] == t_days, "q_implied"].iloc[0]
        includes_ex = (t_ex_days is not None) and (0 <= t_ex_days <= t_days)
        unreliable = t_days < 5
        summary_rows.append(
            {
                "T_days": int(t_days),
                "n": int(len(g)),
                "includes_ex": bool(includes_ex),
                "mean_gap": float(mean_gap),
                "std": float(std_gap),
                "mean_bps": float(mean_bps),
                "frac_tradable": float(frac_tradable),
                "q_impl": float(q_impl),
                "unreliable": bool(unreliable),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("T_days").reset_index(drop=True)

    avg_spread = float(df["spread"].mean())
    avg_abs_gap = float(df["diff"].abs().mean())
    pct_within = float((df["diff"].abs() <= df["spread"]).mean() * 100)

    reliable = summary[~summary["unreliable"]]
    if len(reliable) >= 1:
        mean_q_reliable = float(reliable["q_impl"].median())
    else:
        mean_q_reliable = float(implied_q_by_t["q_implied"].mean())

    diagnostics = []
    diagnostics.append(f"Avg abs gap: ${avg_abs_gap:.3f}")
    diagnostics.append(f"Avg spread: ${avg_spread:.3f}")
    diagnostics.append(f"Contracts within spread: {pct_within:.1f}%")
    diagnostics.append(f"Median q_implied (reliable expiries): {mean_q_reliable * 100:+.3f}%")
    diagnostics.append(f"Input q_custom: {q_custom * 100:+.3f}%")
    diagnostics.append(f"Reported dividend yield: {reported_q * 100:+.3f}%")

    return {
        "ticker": ticker,
        "df": df,
        "summary": summary,
        "implied_q_by_t": implied_q_by_t,
        "spot": spot,
        "yahoo_spot": yahoo_spot,
        "spot_drift": spot_drift,
        "reported_q": reported_q,
        "market_open": market_open,
        "now_et": now_et,
        "stale_flag": stale_flag,
        "quote_age_hours": quote_age_hours,
        "t_ex_days": t_ex_days,
        "next_ex_date": next_ex_date,
        "next_div_amount": next_div_amount,
        "r_rate": r_rate,
        "q_custom": q_custom,
        "chain_counts": pd.DataFrame(chain_counts),
        "avg_spread": avg_spread,
        "avg_abs_gap": avg_abs_gap,
        "pct_within": pct_within,
        "diagnostics": diagnostics,
    }


def build_figure(result: dict) -> plt.Figure:
    df = result["df"]
    summary = result["summary"]
    ticker = result["ticker"]
    spot = result["spot"]
    r_rate = result["r_rate"]
    q_custom = result["q_custom"]
    reported_q = result["reported_q"]
    t_ex_days = result["t_ex_days"]
    next_ex_date = result["next_ex_date"]
    next_div_amount = result["next_div_amount"]

    bg = "#0B0E14"
    fg = "#E6EDF3"
    dim = "#8B949E"
    gold = "#FFB900"
    blue = "#58A6FF"
    green = "#3FB950"
    red = "#F85149"
    purple = "#BC8CFF"

    plt.rcParams.update(
        {
            "figure.facecolor": bg,
            "axes.facecolor": bg,
            "axes.edgecolor": dim,
            "axes.labelcolor": dim,
            "xtick.color": dim,
            "ytick.color": dim,
            "text.color": fg,
            "axes.titlecolor": fg,
            "grid.color": "#21262D",
            "grid.alpha": 0.5,
            "font.family": "DejaVu Sans",
        }
    )

    fig = plt.figure(figsize=(16, 13.5))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.28, height_ratios=[1, 1, 1.15])

    div_label = f"q={q_custom * 100:.2f}%"
    if next_div_amount > 0 and next_ex_date:
        div_label += f" + D=${next_div_amount:.2f} on {next_ex_date}"

    fig.suptitle(
        f"Synthetic Forwards vs Cost-of-Carry - {ticker}  Spot ${spot:.2f}  r={r_rate*100:.1f}%  {div_label}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    def clean(ax, title, subtitle=""):
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=18)
        if subtitle:
            ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=dim, style="italic", va="bottom")
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    best_t = int(df.groupby("T_days").size().idxmax())
    sub = df[df["T_days"] == best_t].sort_values("K")

    ax = fig.add_subplot(gs[0, 0])
    clean(ax, f"1) Construction check - T = {best_t} days")
    f_act_val = sub["F_actual"].iloc[0]
    f_act_noq = sub["F_actual_noq"].iloc[0]
    pv_d = sub["PV_div"].iloc[0]
    if pv_d > 0:
        blue_label = f"F_actual = (S-PV_div)*exp(rT) = ${f_act_val:.3f}"
    else:
        blue_label = f"F_actual = S*exp((r-q)T) = ${f_act_val:.3f}"
    ax.axhline(f_act_val, color=blue, lw=2, zorder=2, label=blue_label)
    ax.axhline(f_act_noq, color=red, lw=1.2, ls=":", alpha=0.7, zorder=1, label=f"No-div baseline = ${f_act_noq:.3f}")
    ax.scatter(sub["K"], sub["F_syn"], s=60, color=gold, edgecolor=bg, lw=0.6, zorder=3, label="F_syn")
    ymin = min(sub["F_syn"].min(), f_act_val, f_act_noq)
    ymax = max(sub["F_syn"].max(), f_act_val, f_act_noq)
    pad = (ymax - ymin) * 0.3 + 0.1
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("Strike K ($)")
    ax.set_ylabel("Forward price ($)")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    ax = fig.add_subplot(gs[0, 1])
    clean(ax, "2) Gap across strikes, by expiry")
    cmap = plt.get_cmap("plasma")
    t_list = sorted(df["T_days"].unique())
    for t_days, col in zip(t_list, cmap(np.linspace(0.15, 0.85, len(t_list)))):
        g = df[df["T_days"] == t_days].sort_values("K")
        ax.plot(g["K"], g["diff_bps"], "o-", color=col, lw=1.5, ms=5, alpha=0.9, label=f"{int(t_days)}d")
    ax.axhline(0, color=fg, lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Strike K ($)")
    ax.set_ylabel("Gap (bps of F)")
    ax.legend(fontsize=8, ncol=2, title="Maturity", title_fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    clean(ax, "3) Mean gap vs maturity")
    g = (
        df.groupby("T_days")
        .agg(mean=("diff", "mean"), std=("diff", "std"), spread=("spread", "mean"))
        .reset_index()
    )
    g["std"] = g["std"].fillna(0)
    ax.fill_between(g["T_days"], -g["spread"], g["spread"], color=gold, alpha=0.2, label="+/- 1 spread")
    ax.errorbar(g["T_days"], g["mean"], yerr=g["std"], fmt="o-", color=blue, capsize=4, lw=1.8, ms=8, label="Mean gap +/- 1sd")
    ax.axhline(0, color=fg, lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel("Gap ($)")
    ax.legend(fontsize=9, loc="best")

    ax = fig.add_subplot(gs[1, 1])
    clean(ax, "4) Tradability check: |gap| vs spread")
    sc = ax.scatter(df["diff"].abs(), df["spread"], c=df["T_days"], cmap="plasma", s=32, alpha=0.8, edgecolor=bg, lw=0.3)
    mx = max(df["diff"].abs().quantile(0.98), df["spread"].quantile(0.98)) * 1.15
    ax.plot([0, mx], [0, mx], ls="--", color=fg, lw=1, alpha=0.6, label="|gap| = spread")
    xx = np.linspace(0, mx, 100)
    ax.fill_between(xx, xx, mx, color=green, alpha=0.1, label="Friction dominated")
    ax.fill_between(xx, 0, xx, color=red, alpha=0.1, label="Potentially tradable")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_xlabel("|Gap| ($)")
    ax.set_ylabel("Bid-ask spread ($)")
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Days to expiry", color=dim, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=dim)
    ax.legend(fontsize=8, loc="lower right")

    ax = fig.add_subplot(gs[2, :])
    clean(ax, "5) q_implied by expiry with ex-date split")
    plot_df = summary[~summary["unreliable"]].copy().sort_values("T_days")

    if len(plot_df) > 0:
        pre_ex = plot_df[~plot_df["includes_ex"]]
        post_ex = plot_df[plot_df["includes_ex"]]

        ax.plot(plot_df["T_days"], plot_df["q_impl"] * 100, color=dim, lw=1, ls=":", alpha=0.5, zorder=1)

        if len(pre_ex):
            ax.scatter(pre_ex["T_days"], pre_ex["q_impl"] * 100, s=120, color=blue, edgecolor=fg, lw=1.1, zorder=3, label="Ex-date outside")

        if len(post_ex):
            ax.scatter(
                post_ex["T_days"],
                post_ex["q_impl"] * 100,
                s=120,
                color=gold,
                edgecolor=fg,
                lw=1.1,
                marker="D",
                zorder=3,
                label="Ex-date inside",
            )

        if t_ex_days is not None:
            ax.axvline(t_ex_days, color=red, lw=2, ls="--", alpha=0.7, zorder=2)
            ylim_now = ax.get_ylim()
            ax.text(
                t_ex_days,
                ylim_now[0] + (ylim_now[1] - ylim_now[0]) * 0.05,
                f" ex-date ({t_ex_days}d)",
                color=red,
                fontsize=10,
                fontweight="bold",
                va="bottom",
                ha="left",
            )

        if len(pre_ex) and len(post_ex):
            pre_med = pre_ex["q_impl"].median() * 100
            post_med = post_ex["q_impl"].median() * 100
            step = post_med - pre_med
            arrow_x = t_ex_days + 1.5 if t_ex_days else (pre_ex["T_days"].max() + post_ex["T_days"].min()) / 2
            ax.axhline(pre_med, color=blue, lw=1, ls="-", alpha=0.4, zorder=1)
            ax.axhline(post_med, color=gold, lw=1, ls="-", alpha=0.4, zorder=1)
            ax.annotate("", xy=(arrow_x, post_med), xytext=(arrow_x, pre_med), arrowprops=dict(arrowstyle="<->", color=green, lw=2.2))
            ax.text(arrow_x + 0.8, (pre_med + post_med) / 2, f"Step: +{step:.2f}%", color=green, fontsize=10, fontweight="bold", va="center")

        ax.axhline(reported_q * 100, color=purple, lw=1, ls=":", alpha=0.7, label=f"Reported q = {reported_q * 100:.2f}%")

    ax.set_xlabel("Days to expiry")
    ax.set_ylabel("q_implied (%)")
    ax.legend(fontsize=9, loc="upper left")

    below = (df["diff"].abs() <= df["spread"]).mean() * 100
    fig.text(
        0.5,
        0.005,
        f"Parity within spread for {below:.0f}% of contracts | Residual gap can come from dividends, borrow, funding, stale quotes",
        ha="center",
        color=dim,
        fontsize=9.5,
        style="italic",
    )

    return fig


def main() -> None:
    st.set_page_config(page_title="Synthetic Forward Analyzer", layout="wide")
    st.title("Synthetic Forward Analysis")
    st.caption("Interactive parity diagnostics for option-implied forward pricing.")

    with st.sidebar:
        st.header("Inputs")
        ticker = st.text_input("Ticker", value="AAPL")
        r_rate = st.number_input("Risk-free rate r", min_value=-0.05, max_value=0.30, value=0.045, step=0.005, format="%.4f")
        q_custom = st.number_input("Custom continuous yield q", min_value=-0.10, max_value=0.30, value=0.0, step=0.0025, format="%.4f")

        st.subheader("Dividend model")
        use_discrete_div = st.checkbox("Use discrete dividend", value=True)
        next_div_amount = st.number_input("Next dividend amount ($)", min_value=0.0, max_value=25.0, value=0.26, step=0.01)
        ex_date = st.date_input("Next ex-dividend date", value=date.today())

        st.subheader("Filters")
        min_oi = st.slider("Minimum open interest", min_value=0, max_value=500, value=20, step=5)
        atm_window = st.slider("ATM window (+/- % of spot)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
        max_expiries = st.slider("Max expiries", min_value=1, max_value=20, value=10, step=1)

        run_now = st.button("Run analysis", type="primary", use_container_width=True)

    if "last_params" not in st.session_state:
        st.session_state["last_params"] = None

    if run_now:
        ex_date_str = ex_date.strftime("%Y-%m-%d") if use_discrete_div else None
        div_amount = float(next_div_amount) if use_discrete_div else 0.0
        st.session_state["last_params"] = {
            "ticker": ticker,
            "r_rate": float(r_rate),
            "q_custom": float(q_custom),
            "next_div_amount": div_amount,
            "next_ex_date": ex_date_str,
            "min_oi": int(min_oi),
            "atm_window": float(atm_window),
            "max_expiries": int(max_expiries),
        }

    if st.session_state["last_params"] is None:
        st.info("Pick inputs and click Run analysis.")
        return

    params = st.session_state["last_params"]

    try:
        with st.spinner("Loading option chains and computing diagnostics..."):
            result = run_analysis(**params)
            fig = build_figure(result)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot used", f"${result['spot']:.2f}")
    c2.metric("Yahoo spot", f"${result['yahoo_spot']:.2f}", delta=f"{result['spot_drift']:+.2f}")
    c3.metric("Contracts within spread", f"{result['pct_within']:.1f}%")
    market_state = "Open" if result["market_open"] else "Closed"
    c4.metric("US options market", market_state)

    if result["stale_flag"]:
        st.warning("Quotes look stale (older than 2 hours while market is closed). Results are indicative.")

    if result["quote_age_hours"] is not None:
        st.caption(f"Latest quote age: {result['quote_age_hours']:.1f} hours | ET approx: {result['now_et'].strftime('%Y-%m-%d %H:%M')}")

    st.pyplot(fig, use_container_width=True)

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
    png_buf.seek(0)

    csv_bytes = result["df"].to_csv(index=False).encode("utf-8")

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download contracts CSV",
        data=csv_bytes,
        file_name=f"synthetic_forward_{result['ticker']}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "Download diagnostics figure PNG",
        data=png_buf,
        file_name=f"synthetic_forward_{result['ticker']}.png",
        mime="image/png",
        use_container_width=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Expiry Summary", "Contracts", "Diagnostics", "Chain Filter Counts"])

    with tab1:
        show_summary = result["summary"].copy()
        show_summary["includes_ex"] = show_summary["includes_ex"].map(lambda x: "Yes" if x else "No")
        show_summary["unreliable"] = show_summary["unreliable"].map(lambda x: "Yes" if x else "No")
        show_summary["q_impl_pct"] = show_summary["q_impl"] * 100
        st.dataframe(show_summary, use_container_width=True)

    with tab2:
        cols = [
            "expiry",
            "T_days",
            "K",
            "C",
            "P",
            "spread",
            "F_syn",
            "F_actual",
            "diff",
            "diff_bps",
        ]
        st.dataframe(result["df"][cols].sort_values(["T_days", "K"]), use_container_width=True)

    with tab3:
        for line in result["diagnostics"]:
            st.write(f"- {line}")

        if result["next_div_amount"] > 0 and result["t_ex_days"] is not None:
            st.write(
                f"- Discrete dividend modeled: ${result['next_div_amount']:.2f} on {result['next_ex_date']} (T_ex={result['t_ex_days']} days)."
            )
        else:
            st.write("- Discrete dividend not modeled.")

    with tab4:
        st.dataframe(result["chain_counts"], use_container_width=True)


if __name__ == "__main__":
    main()
