from __future__ import annotations
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from dashboard_pipeline import (  # noqa: E402
    _ALL_INPUT_FEATURES,
    build_training_bundle,
    cost_sensitivity,
    cost_sensitivity_by_model,
    evaluate_at_profits,
    focused_comparison_rows,
    probability_shap_explanation,
    recommended_model_row,
    input_field_hints,
    score_customers,
    shap_explanation,
)
from chatbot import answer_chatbot_question, build_chat_context  # noqa: E402
from dashboard_visuals import (  # noqa: E402
    PALETTE,
    confusion_matrix_figure,
    expected_profit_by_model,
    focused_model_comparison,
    model_ratio_curves,
    render_probability_waterfall_plot,
    render_shap_dependence_plot,
    render_shap_summary_plot,
)


st.set_page_config(
    page_title="Bank Telemarketing Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_style() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, #f8fbfd 0%, {PALETTE["paper"]} 100%);
            color: {PALETTE["ink"]};
        }}
        .block-container {{
            padding-top: 2.8rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }}
        h1, h2, h3 {{
            color: {PALETTE["ink"]};
            letter-spacing: 0;
        }}
        p, label, span {{
            color: {PALETTE["ink"]};
        }}
        [data-testid="stMarkdownContainer"] p {{
            color: {PALETTE["ink"]};
            line-height: 1.55;
        }}
        [data-testid="stWidgetLabel"] p {{
            color: {PALETTE["ink"]} !important;
            font-weight: 700;
        }}
        [data-testid="stMetric"] {{
            background: {PALETTE["panel"]};
            border: 1px solid {PALETTE["line"]};
            border-radius: 8px;
            padding: 0.95rem 1rem;
            min-height: 126px;
            box-shadow: 0 10px 28px rgba(16, 32, 51, 0.07);
        }}
        [data-testid="stMetricLabel"] p {{
            color: {PALETTE["muted"]} !important;
            font-weight: 700;
        }}
        [data-testid="stMetricValue"] {{
            color: {PALETTE["ink"]};
            font-size: 1.55rem;
            font-weight: 800;
        }}
        [data-testid="stMetricDelta"] {{
            font-size: 0.88rem;
        }}
        [data-testid="stMetricDelta"] svg {{
            fill: {PALETTE["teal"]} !important;
        }}
        [data-testid="stMetricDelta"] div {{
            color: {PALETTE["teal"]} !important;
        }}
        [data-baseweb="tab-list"] {{
            gap: 0.8rem;
            margin-bottom: 0.25rem;
        }}
        [data-baseweb="tab-list"] button {{
            cursor: pointer;
            border: 1px solid rgba(83, 101, 122, 0.18);
            border-radius: 10px;
            padding: 0.6rem 1rem 0.7rem;
            min-height: 2.9rem;
            background: rgba(255, 255, 255, 0.72);
            transition: background-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }}
        [data-baseweb="tab-list"] button:hover {{
            background: rgba(15, 118, 110, 0.1);
            border-color: rgba(15, 118, 110, 0.24);
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(16, 32, 51, 0.08);
        }}
        [data-baseweb="tab-list"] button[aria-selected="true"] {{
            background: rgba(15, 118, 110, 0.14);
            border-color: rgba(15, 118, 110, 0.3);
            box-shadow: inset 0 -3px 0 0 {PALETTE["teal"]}, 0 8px 18px rgba(16, 32, 51, 0.06);
        }}
        [data-baseweb="tab-list"] button p {{
            color: {PALETTE["ink"]} !important;
            font-weight: 850;
            letter-spacing: 0.01rem;
            font-size: 1.02rem;
        }}
        [data-baseweb="tab-highlight"] {{
            background-color: {PALETTE["teal"]} !important;
            height: 3px !important;
        }}
        [data-baseweb="slider"] [role="slider"] {{
            background-color: {PALETTE["teal"]} !important;
        }}
        [data-baseweb="slider"] div {{
            color: {PALETTE["ink"]};
        }}
        [data-testid="stForm"] {{
            background: {PALETTE["panel"]};
            border: 1px solid {PALETTE["line"]};
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 14px 38px rgba(16, 32, 51, 0.08);
        }}
        .hero {{
            background: linear-gradient(135deg, {PALETTE["navy"]} 0%, {PALETTE["navy_2"]} 70%, #214f60 100%);
            border-radius: 8px;
            padding: 1.35rem 1.7rem;
            margin-bottom: 1rem;
            border: 1px solid #284862;
            box-shadow: 0 16px 36px rgba(16, 32, 51, 0.16);
        }}
        .hero h1 {{
            color: #ffffff !important;
            margin: 0.2rem 0 0.45rem;
            font-size: 2.28rem;
        }}
        .hero h1 span {{
            color: #ffffff !important;
        }}
        .hero p {{
            color: #f1f7fb !important;
            max-width: 760px;
            font-size: 0.98rem;
            margin-bottom: 0;
        }}
        .hero-kicker {{
            color: #9fe7df;
            text-transform: uppercase;
            letter-spacing: 0.08rem;
            font-size: 0.78rem;
            font-weight: 800;
        }}
        .home-intro {{
            color: {PALETTE["ink"]};
            line-height: 1.45;
            margin: 0 0 0.75rem;
        }}
        .home-card {{
            background: {PALETTE["panel"]};
            border: 1px solid {PALETTE["line"]};
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 30px rgba(16, 32, 51, 0.07);
        }}
        .home-card strong {{
            display: block;
            color: {PALETTE["ink"]};
            margin-bottom: 0.55rem;
            font-size: 1rem;
        }}
        .home-card span {{
            display: block;
            color: {PALETTE["muted"]};
            padding: 0.2rem 0;
            font-weight: 650;
        }}
        .home-visual {{
            background:
                linear-gradient(135deg, rgba(16, 32, 51, 0.96), rgba(29, 53, 80, 0.92)),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.05) 0, rgba(255,255,255,0.05) 1px, transparent 1px, transparent 34px);
            border: 1px solid #284862;
            border-radius: 8px;
            padding: 1rem;
            min-height: 250px;
            box-shadow: 0 18px 42px rgba(16, 32, 51, 0.16);
        }}
        .home-visual strong {{
            display: block;
            color: #ffffff;
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }}
        .home-visual span {{
            color: #d8e8ef;
            font-size: 0.86rem;
        }}
        .visual-bars {{
            display: flex;
            align-items: flex-end;
            gap: 0.6rem;
            height: 112px;
            margin-top: 1.05rem;
            border-bottom: 1px solid rgba(255,255,255,0.24);
        }}
        .visual-bar {{
            width: 18%;
            border-radius: 6px 6px 0 0;
            background: linear-gradient(180deg, #9fe7df, {PALETTE["teal"]});
            box-shadow: 0 8px 18px rgba(15, 118, 110, 0.28);
        }}
        .visual-bar.secondary {{
            background: linear-gradient(180deg, #8fb9df, {PALETTE["blue"]});
        }}
        .visual-labels {{
            display: flex;
            justify-content: space-between;
            margin-top: 0.7rem;
        }}
        .formula-card {{
            background: linear-gradient(180deg, #ffffff 0%, #f7fbfb 100%);
            border: 1px solid #c7dfe0;
            border-radius: 8px;
            padding: 0.82rem 0.95rem;
            margin-top: 0.45rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 12px 26px rgba(16, 32, 51, 0.07);
        }}
        .formula-title {{
            display: block;
            color: {PALETTE["muted"]};
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06rem;
            margin-bottom: 0.52rem;
        }}
        .formula-row {{
            display: flex;
            flex-wrap: nowrap;
            align-items: center;
            gap: 0.34rem;
            overflow-x: auto;
            padding-bottom: 0.1rem;
        }}
        .formula-token {{
            display: inline-flex;
            align-items: center;
            min-height: 2rem;
            border-radius: 8px;
            padding: 0.35rem 0.52rem;
            background: #eef7f6;
            border: 1px solid #bedbd7;
            color: {PALETTE["ink"]};
            font-weight: 800;
            white-space: nowrap;
            font-size: 0.92rem;
        }}
        .formula-token.result {{
            background: {PALETTE["navy"]};
            border-color: {PALETTE["navy"]};
            color: #ffffff;
        }}
        .formula-operator {{
            color: {PALETTE["teal"]};
            font-weight: 900;
            font-size: 1rem;
        }}
        .formula-note {{
            color: {PALETTE["muted"]};
            font-size: 0.84rem;
            margin-top: 0.5rem;
        }}
        .loading-card {{
            background: {PALETTE["panel"]};
            border: 1px solid {PALETTE["line"]};
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 14px 38px rgba(16, 32, 51, 0.08);
        }}
        .loading-card strong {{
            display: block;
            color: {PALETTE["ink"]};
            font-size: 1.05rem;
            margin-bottom: 0.35rem;
        }}
        .loading-card span {{
            color: {PALETTE["muted"]};
            display: block;
            line-height: 1.45;
        }}
        .loading-visual {{
            background:
                linear-gradient(135deg, rgba(16, 32, 51, 0.98), rgba(29, 53, 80, 0.94)),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.05) 0, rgba(255,255,255,0.05) 1px, transparent 1px, transparent 34px);
            border: 1px solid #284862;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 0.75rem;
            box-shadow: 0 14px 36px rgba(16, 32, 51, 0.15);
        }}
        .loading-visual strong {{
            display: block;
            color: #ffffff;
            font-size: 1rem;
            margin-bottom: 0.75rem;
        }}
        .loading-rail {{
            height: 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.15);
            overflow: hidden;
        }}
        .loading-rail span {{
            display: block;
            height: 100%;
            width: 42%;
            border-radius: 999px;
            background: linear-gradient(90deg, #9fe7df, {PALETTE["teal"]}, #8fb9df);
            animation: loadingSweep 1.45s ease-in-out infinite;
        }}
        .loading-steps {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.9rem;
        }}
        .loading-step {{
            border: 1px solid rgba(159,231,223,0.28);
            background: rgba(255,255,255,0.06);
            border-radius: 8px;
            padding: 0.65rem;
            color: #e6f3f6;
            font-weight: 750;
            text-align: center;
            font-size: 0.85rem;
        }}
        @keyframes loadingSweep {{
            0% {{ transform: translateX(-110%); }}
            50% {{ transform: translateX(35%); }}
            100% {{ transform: translateX(245%); }}
        }}
        [data-testid="stNumberInput"] input {{
            color: #ffffff !important;
            font-weight: 700;
        }}
        [data-testid="stNumberInput"] button {{
            color: #ffffff !important;
        }}
        [data-testid="stTextInput"] input {{
            color: #ffffff !important;
            font-weight: 700;
        }}
        [data-testid="stTextInput"] input::placeholder {{
            color: #c9d3df !important;
            opacity: 1;
        }}
        .section-note {{
            color: {PALETTE["muted"]};
            max-width: 920px;
            margin-bottom: 1rem;
            line-height: 1.45;
            font-size: 1.1rem;
        }}
        .muted {{
            color: {PALETTE["muted"]};
        }}
        .status-pill {{
            display: inline-block;
            border: 1px solid #b7d4d1;
            border-radius: 999px;
            padding: 0.24rem 0.68rem;
            font-size: 0.82rem;
            color: #123d46;
            background: #e9f6f4;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }}
        .winner-panel {{
            background: {PALETTE["panel"]};
            border: 1px solid {PALETTE["line"]};
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin: 0.35rem 0 1rem;
            box-shadow: 0 10px 26px rgba(16, 32, 51, 0.07);
        }}
        .winner-panel strong {{
            display: block;
            color: {PALETTE["ink"]};
            font-size: 1.08rem;
            margin-bottom: 0.7rem;
        }}
        .winner-pill {{
            display: inline-block;
            border: 1px solid #93c9c3;
            border-radius: 8px;
            padding: 0.72rem 1rem;
            min-width: 180px;
            text-align: center;
            color: #0f2f3a;
            background: linear-gradient(180deg, #f5fbfa 0%, #e1f3f1 100%);
            margin-right: 0.7rem;
            margin-bottom: 0.55rem;
            font-size: 1.05rem;
            font-weight: 800;
        }}
        div.stButton > button,
        [data-testid="stFormSubmitButton"] button {{
            border-radius: 8px;
            border: 1px solid {PALETTE["line"]};
            background: {PALETTE["panel"]} !important;
            color: {PALETTE["ink"]} !important;
            font-weight: 700;
            box-shadow: 0 8px 18px rgba(16, 32, 51, 0.06);
        }}
        div.stButton > button p,
        [data-testid="stFormSubmitButton"] button p {{
            color: {PALETTE["ink"]} !important;
            font-weight: 700;
        }}
        [data-testid="stBaseButton-primary"],
        [data-testid="stFormSubmitButton"] button {{
            background: linear-gradient(135deg, {PALETTE["teal"]} 0%, #155e75 100%) !important;
            border-color: {PALETTE["teal"]} !important;
            color: #ffffff !important;
        }}
        [data-testid="stBaseButton-primary"] p,
        [data-testid="stFormSubmitButton"] button p {{
            color: #ffffff !important;
            font-weight: 800;
        }}
        [data-testid="stBaseButton-secondary"] {{
            background: {PALETTE["panel"]} !important;
            color: {PALETTE["ink"]} !important;
        }}
        .dashboard-header {{
            padding-top: 0.4rem;
        }}
        .dashboard-header [data-testid="stVerticalBlock"] {{
            overflow: visible;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def cached_bundle():
    return build_training_bundle()


@st.cache_resource(show_spinner=False)
def dashboard_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_shap_explanation(_bundle, model_name: str, sample_size: int):
    """Cache the global SHAP explanation per (model, sample_size).
 
    The bundle is prefixed with an underscore so Streamlit does not try to
    hash it; model_name and sample_size fully determine the result given a
    fixed trained bundle and RANDOM_STATE-based sampling.
    """
    return shap_explanation(_bundle, model_name, sample_size=sample_size)
 
 
@st.cache_data(show_spinner=False, max_entries=64)
def cached_probability_shap_explanation(_bundle, model_name: str, input_key: str, input_df: pd.DataFrame):
    """Cache the per-customer probability SHAP explanation.
 
    input_key is a stable string fingerprint of the customer's raw input so
    identical inputs reuse the cached SHAP result instead of recomputing on
    every click.
    """
    return probability_shap_explanation(_bundle, model_name, input_df)


def money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def html_money(value: float) -> str:
    return money(value).replace("$", "&#36;")


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def signed_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:+.1%}"


def savings_pct(savings: float, baseline: float) -> float:
    if baseline == 0 or pd.isna(savings) or pd.isna(baseline):
        return float("nan")
    return savings / baseline


def set_page(page: str) -> None:
    st.session_state["app_page"] = page
    st.session_state["dashboard_loading"] = page == "loading"
    st.session_state["dashboard_ready"] = page == "dashboard"


def render_home_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Bank campaign analytics</div>
          <h1 style="color:#ffffff !important;">Bank Telemarketing Optimizer</h1>
          <p style="color:#f1f7fb !important;">
            Choose campaign revenue and cost assumptions, identify the highest-profit model,
            compare it against practical baselines, test sensitivity of revenue-to-cost ratios
            and inspect the global drivers and customer level reasons behind a call decision.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profit_formula() -> None:
    st.markdown(
        """
        <div class="formula-card">
          <span class="formula-title">Business Profit Function</span>
          <div class="formula-row">
            <span class="formula-token result">Expected Profit</span>
            <span class="formula-operator">=</span>
            <span class="formula-token">True Positives</span>
            <span class="formula-operator">×</span>
            <span class="formula-token">(Revenue - Cost of Call)</span>
            <span class="formula-operator">−</span>
            <span class="formula-token">False Positives</span>
            <span class="formula-operator">×</span>
            <span class="formula-token">Cost of Call</span>
          </div>
          <div class="formula-note">The dashboard ranks models by the highest expected net profit under your assumptions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_visual() -> None:
    st.markdown(
        """
        <div class="home-visual">
          <strong>Campaign Profit Lens</strong>
          <span>Compare model-led targeting against broad outreach to maximize marketing net profit.</span>
          <div class="visual-bars">
            <div class="visual-bar secondary" style="height: 30%;"></div>
            <div class="visual-bar secondary" style="height: 42%;"></div>
            <div class="visual-bar" style="height: 60%;"></div>
            <div class="visual-bar" style="height: 78%;"></div>
            <div class="visual-bar" style="height: 92%;"></div>
          </div>
          <div class="visual-labels"><span>Baseline</span><span>Optimized</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def home_screen() -> None:
    render_home_hero()

    intro_left, intro_right = st.columns([0.68, 0.32], gap="large")
    with intro_left:
        render_profit_formula()
        with st.form("profit_form"):
            c1, c2 = st.columns(2)
            cost_fn_text = c1.text_input(
                "Estimated subscription revenue ($): calling a customer who would subscribe",
                placeholder="Enter Estimated Subscription Revenue",
            )
            cost_fp_text = c2.text_input(
                "Cost of call ($): calling a customer regardless of their interest",
                placeholder="Enter Call Cost",
            )
            submitted = st.form_submit_button("Build dashboard", type="primary", width="stretch")
    with intro_right:
        render_home_visual()
    if submitted:
        try:
            cost_fp = float(cost_fp_text)
            cost_fn = float(cost_fn_text)
        except ValueError:
            st.warning("Please enter valid numeric cost values before building the dashboard.")
            return
        if cost_fp <= 0 or cost_fn <= 0:
            st.warning("Please enter positive cost values before building the dashboard.")
            return

        st.session_state["pending_cost_fp"] = float(cost_fp)
        st.session_state["pending_cost_fn"] = float(cost_fn)
        st.session_state.pop("dashboard_results", None)
        st.session_state.pop("dashboard_best", None)
        st.session_state.pop("dashboard_cost_key", None)
        st.session_state.pop("dashboard_bundle", None)
        st.session_state.pop("dashboard_future", None)
        st.session_state.pop("dashboard_future_key", None)
        set_page("loading")
        st.rerun()
        st.stop()


def render_loading_home_content(cost_fp: float, cost_fn: float) -> None:
    render_home_hero()

    intro_left, intro_right = st.columns([0.68, 0.32], gap="large")
    with intro_left:
        st.markdown(
            f"""
            <div class="loading-card">
              <strong>Preparing your campaign decision dashboard</strong>
              <span>Using call cost {html_money(cost_fp)} and estimated subscription revenue {html_money(cost_fn)}.</span>
              <span>Loading saved models, optimizing thresholds, and calculating expected profits.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_profit_formula()
        st.markdown(
            """
            <div class="loading-visual">
              <strong>Building dashboard...</strong>
              <div class="loading-rail"><span></span></div>
              <div class="loading-steps">
                <div class="loading-step">Load models</div>
                <div class="loading-step">Optimize thresholds</div>
                <div class="loading-step">Calculate profits</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with intro_right:
        render_home_visual()


def loading_home_screen() -> None:
    cost_fp = float(st.session_state["pending_cost_fp"])
    cost_fn = float(st.session_state["pending_cost_fn"])
    render_loading_home_content(cost_fp, cost_fn)

    future: Future | None = st.session_state.get("dashboard_future")
    future_key = st.session_state.get("dashboard_future_key")
    cost_key = (cost_fp, cost_fn)
    if future is None or future_key != cost_key:
        st.session_state["dashboard_future_key"] = cost_key
        st.session_state["dashboard_future"] = dashboard_executor().submit(prepare_dashboard_state, cost_fp, cost_fn)
    loading_status_fragment()


def hide_loading_artifacts_on_dashboard() -> None:
    st.markdown(
        """
        <style>
        .loading-card,
        .loading-visual {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
            overflow: hidden !important;
            opacity: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_results(bundle, cost_fp: float, cost_fn: float) -> pd.DataFrame:
    return evaluate_at_profits(bundle, cost_fp=cost_fp, cost_fn=cost_fn)


def prepare_dashboard_state(cost_fp: float, cost_fn: float, bundle=None):
    bundle = bundle if bundle is not None else build_training_bundle()
    results = get_results(bundle, cost_fp, cost_fn)
    best = recommended_model_row(results)
    return bundle, results, best


@st.fragment(run_every="1s")
def loading_status_fragment() -> None:
    future: Future | None = st.session_state.get("dashboard_future")
    cost_fp = float(st.session_state["pending_cost_fp"])
    cost_fn = float(st.session_state["pending_cost_fn"])
    if future is None:
        return

    if not future.done():
        st.info("Loading saved models and calculating the best campaign strategy...")
        return

    try:
        bundle, results, best = future.result()
    except Exception as exc:
        set_page("home")
        st.session_state.pop("dashboard_future", None)
        st.session_state.pop("dashboard_future_key", None)
        st.error(f"Could not prepare the dashboard: {exc}")
        return

    if best is None:
        set_page("home")
        st.session_state.pop("dashboard_future", None)
        st.session_state.pop("dashboard_future_key", None)
        st.error("No models trained successfully. Check the dependency setup and training errors.")
        st.dataframe(results, width='stretch', hide_index=True)
        return

    st.session_state["cost_fp"] = cost_fp
    st.session_state["cost_fn"] = cost_fn
    st.session_state["dashboard_bundle"] = bundle
    st.session_state["dashboard_results"] = results
    st.session_state["dashboard_best"] = best
    st.session_state["dashboard_cost_key"] = (cost_fp, cost_fn)
    st.session_state.pop("dashboard_future", None)
    st.session_state.pop("dashboard_future_key", None)
    set_page("dashboard")
    st.rerun(scope="app")


def overview_tab(results: pd.DataFrame, best: dict, cost_fp: float, cost_fn: float) -> None:
    st.subheader("Recommended Decision Strategy")
    st.markdown(
        '<div class="section-note">'
        "The selected model is the one with the highest expected profit on the held-out test set. "
        "The profits are compared against a a call-everyone campaign and logistic-regression baseline."
        "</div>",
        unsafe_allow_html=True,
    )
    call_all_pct = savings_pct(best["profit_gain_vs_call_everyone"], best["call_everyone_profit"])
    logistic_pct = savings_pct(best["profit_gain_vs_logistic"], best["logistic_profit"])
    cols = st.columns(4)
    cols[0].metric("Recommended Model", best["model"])
    cols[1].metric("Expected Profit", money(best["expected_profit"]), "Higher is better")
    cols[2].metric("Profit Gain vs Call Everyone", money(best["profit_gain_vs_call_everyone"]), signed_pct(call_all_pct))
    cols[3].metric("Profit Gain vs Logistic Regression", money(best["profit_gain_vs_logistic"]), signed_pct(logistic_pct))

    st.markdown(
        f"""
        <span class="status-pill">Call Cost {html_money(cost_fp)}</span>
        <span class="status-pill">Estimated Subscription Revenue {html_money(cost_fn)}</span>
        <span class="status-pill">Revenue-to-Cost Ratio {cost_fn / cost_fp:.2f}</span>
        <span class="status-pill">Recall {best['recall']:.1%}</span>
        <span class="status-pill">Precision {best['precision']:.1%}</span>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.plotly_chart(expected_profit_by_model(results), width='stretch')
    with right:
        st.plotly_chart(confusion_matrix_figure(best), width='stretch')

    st.info(
        f"At current assumptions, the potential revenue from a single subscriber is {html_money(cost_fn)} and the cost of a phone call is {html_money(cost_fp)}. "
        "This means the most profitable model will change if the revenue per customer or the operational call costs shift."
    )


def comparison_sensitivity_tab(bundle, results: pd.DataFrame, cost_fp: float, cost_fn: float) -> None:
    st.subheader("Model Comparison and Cost Sensitivity")
    st.markdown(
        '<div class="section-note">'
        "This view keeps the business decision focused: compare the best model against key baselines, "
        "then analyse whether the winning model changes as the revenue-to-call cost ratio changes."
        "</div>",
        unsafe_allow_html=True,
    )
    focused = focused_comparison_rows(results, bundle.y_test, cost_fp, cost_fn)
    if not focused.empty:
        st.plotly_chart(focused_model_comparison(focused), width='stretch')

    failed = results[~results["status"].eq("ok")]
    if not failed.empty:
        st.warning("Some models were unavailable or failed during training.")
        st.dataframe(failed[["model", "status", "error"]], width='stretch', hide_index=True)

    st.markdown("**Cost Sensitivity: Top Winners Only**")
    sensitivity = cost_sensitivity(bundle, cost_fp=cost_fp)
    sensitivity_detail = cost_sensitivity_by_model(bundle, cost_fp=cost_fp)
    if sensitivity.empty:
        st.error("No sensitivity results are available.")
        return

    winner_models = list(sensitivity["best_model"].drop_duplicates())
    winner_pills = "".join(f'<span class="winner-pill">{model}</span>' for model in winner_models)
    st.markdown(
        f'<div class="winner-panel"><strong>Winners across tested Revenue-to-Cost ratios</strong>{winner_pills}</div>',
        unsafe_allow_html=True,
    )
    st.metric("Most Stable Winner", sensitivity["best_model"].mode().iloc[0])

    st.plotly_chart(model_ratio_curves(sensitivity_detail, winner_models=winner_models), width='stretch')



def explainability_tab(bundle, best: dict) -> None:
    st.subheader("Explainability")
    st.markdown(
        '<div class="section-note">'
        "This view follows the SHAP summary and dependence plot pattern used in the modelling script. "
        "The summary plot shows global feature impact; the dependence plot shows the interaction between the selected feature and the feature with the strongest correlation."
        "</div>",
        unsafe_allow_html=True,
    )
    model_name = st.selectbox(
        "Model to explain",
        options=list(bundle.probabilities.keys()),
        index=list(bundle.probabilities.keys()).index(best["model"]),
    )
    sample_size = st.slider("SHAP sample size", min_value=30, max_value=160, value=80, step=10)
    with st.spinner("Preparing SHAP diagrams..."):
        try:
            explanation, transformed, raw_df = cached_shap_explanation(bundle, model_name, sample_size)
            values = explanation.values
            if getattr(values, "ndim", 0) == 3:
                values = values[:, :, -1]
            top_feature_idx = abs(values).mean(axis=0).argmax()
            top_feature = transformed.columns[top_feature_idx]
            feature_input = st.text_input(
                "Dependence plot x-axis feature",
                value=top_feature,
                help="Type an exact transformed feature name from the SHAP summary plot, such as euribor3m or nr.employed.",
            ).strip()
            if feature_input in transformed.columns:
                dependence_feature = feature_input
            else:
                dependence_feature = top_feature
                st.warning(
                    f"Feature '{feature_input}' was not found in the transformed model features. "
                    f"Showing the strongest feature instead: {top_feature}."
                )
            left, right = st.columns(2, gap="large")
            with left:
                st.markdown("**SHAP Summary Plot**")
                st.caption("Matches the modelling script's global SHAP summary style.")
                render_shap_summary_plot(explanation, transformed, model_name, st, max_display=10)
            with right:
                st.markdown("**SHAP Dependence Plot**")
                st.caption(f"X-axis feature: {dependence_feature} (shown in original units where applicable).")
                render_shap_dependence_plot(explanation, transformed, dependence_feature, model_name, st, raw_df=raw_df)
        except Exception as exc:
            st.error(f"SHAP could not render for this model: {exc}")
            return
 
 
def customer_input_tab(bundle, results: pd.DataFrame, best: dict, cost_fp: float, cost_fn: float) -> None:
    st.subheader("Single Customer Decision")
    st.markdown(
        '<div class="section-note">'
        "Score one customer profile and explain that individual prediction with SHAP waterfall plot. "
        "You can select a specific historical row or enter a new profile manually."
        "</div>",
        unsafe_allow_html=True,
    )
    # raw_test_frame contains the original pre-clean test rows; use it as the
    # source for historical lookups and to populate dropdown options.
    raw_df = bundle.raw_test_frame
    source = st.radio("Input mode", ["Select historical customer", "Manual input"], horizontal=True)
 
    if source == "Select historical customer":
        idx = st.number_input(
            "Specific data point row number",
            min_value=0,
            max_value=len(raw_df) - 1,
            value=0,
            step=1,
            help="Optional: choose an exact row from the test dataset to inspect.",
        )
        idx = int(idx)
        # Use .iloc for positional access — defensive even though
        # build_training_bundle already resets raw_test_frame's index.
        display_cols = [c for c in _ALL_INPUT_FEATURES if c in raw_df.columns]
        input_df = raw_df.iloc[[idx]][display_cols].reset_index(drop=True)
        st.dataframe(input_df, width='stretch', hide_index=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        hints = input_field_hints()
        input_df = pd.DataFrame(
            [
                {
                    "age": c1.number_input("Age", min_value=18, max_value=100, value=42, help=hints["age"]),
                    "job": c2.selectbox("Job", sorted(raw_df["job"].astype(str).unique()), help=hints["job"]),
                    "marital": c3.selectbox("Marital", sorted(raw_df["marital"].astype(str).unique()), help=hints["marital"]),
                    "education": c4.selectbox("Education", sorted(raw_df["education"].astype(str).unique()), help=hints["education"]),
                    "default": 0,
                    "housing": c1.selectbox("Housing loan", sorted(raw_df["housing"].astype(str).unique()), help=hints["housing"]),
                    "loan": c2.selectbox("Personal loan", sorted(raw_df["loan"].astype(str).unique()), help=hints["loan"]),
                    "contact": c3.selectbox("Contact type", sorted(raw_df["contact"].astype(str).unique()), help=hints["contact"]),
                    "month": c4.selectbox("Month", sorted(raw_df["month"].astype(str).unique()), help=hints["month"]),
                    "day_of_week": c1.selectbox("Day", sorted(raw_df["day_of_week"].astype(str).unique()), help=hints["day_of_week"]),
                    "duration": 0,
                    "campaign": c2.number_input("Campaign contacts", min_value=1, max_value=60, value=2, help=hints["campaign"]),
                    "pdays": c3.number_input("Days since previous contact", min_value=0, max_value=999, value=999, help=hints["pdays"]),
                    "previous": c4.number_input("Previous contacts", min_value=0, max_value=10, value=0, help=hints["previous"]),
                    "emp.var.rate": float(raw_df["emp.var.rate"].median()),
                    "cons.price.idx": c1.number_input("Consumer price index", value=float(raw_df["cons.price.idx"].median()), help=hints["cons.price.idx"]),
                    "cons.conf.idx": c2.number_input("Consumer confidence index", value=float(raw_df["cons.conf.idx"].median()), help=hints["cons.conf.idx"]),
                    "euribor3m": c3.number_input("Euribor 3m", value=float(raw_df["euribor3m"].median()), help=hints["euribor3m"]),
                    "nr.employed": c4.number_input("Number employed", value=float(raw_df["nr.employed"].median()), help=hints["nr.employed"]),
                }
            ]
        )
 
    valid_results = results[results["status"].eq("ok")]
    model_options = list(bundle.probabilities.keys())
    model_name = st.selectbox(
        "Decision model",
        options=model_options,
        index=model_options.index(best["model"]),
    )
    model_row = valid_results.loc[valid_results["model"].eq(model_name)]
    threshold = float(model_row.iloc[0]["threshold"]) if not model_row.empty else float(best["threshold"])
    explain_prediction = st.checkbox("Show SHAP waterfall plot for this customer", value=True)
 
    if st.button("Score customer", type="primary"):
        try:
            scored = score_customers(bundle, model_name, input_df, threshold)
            scored_display = scored.copy()
            scored_display.insert(0, "model", model_name)
            st.dataframe(
                scored_display.style.format(
                    {
                        "probability": "{:.1%}",
                        "threshold": "{:.3f}"
                    }
                ),
                width='stretch',
                hide_index=True,
            )
            for _, row in scored.iterrows():
                st.info(
                    f"Decision: {row['decision']} with {row['confidence'].lower()} confidence. "
                    f"Predicted subscription probability is {row['probability']:.1%}."
                )
            if explain_prediction:
                with st.spinner("Building SHAP explanation for this customer..."):
                    customer_row = input_df.iloc[[0]]
                    # Stable fingerprint of the raw input row so identical
                    # inputs reuse the cached SHAP result.
                    input_key = repr(tuple(customer_row.iloc[0].items()))
                    explanation, _transformed, raw_df, probability = cached_probability_shap_explanation(
                        bundle, model_name, input_key, customer_row
                    )
                    single = explanation[0]
                    raw_row = raw_df.iloc[0] if raw_df is not None and not raw_df.empty else None
 
                st.markdown("**SHAP Waterfall Plot**")
                st.caption("Probability-space waterfall, matching the modelling script's customer-level explanation style.")
                render_probability_waterfall_plot(single, probability, model_name, st, max_display=8, raw_row=raw_row)
 
        except Exception as exc:
            st.error(f"Could not score customer input: {exc}")
 


def chatbot_tab(best: dict, cost_fp: float, cost_fn: float) -> None:
    st.subheader("AI Chatbot")
    st.markdown(
        '<div class="section-note">'
        "Ask questions about model choice, thresholds, savings, or customer-level decisions. "
        "</div>",
        unsafe_allow_html=True,
    )
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
 
    prompt = st.chat_input("Ask about models, costs, thresholds, drivers, or a decision...")
    context = build_chat_context(best, cost_fp, cost_fn)
    if prompt:
        history = [
            {"role": role, "content": message}
            for role, message in st.session_state["chat_history"]
        ]
        answer = answer_chatbot_question(prompt, context, history=history)
        st.session_state["chat_history"].append(("user", prompt))
        st.session_state["chat_history"].append(("assistant", answer))
 
    for role, message in reversed(st.session_state["chat_history"]):
        with st.chat_message(role):
            st.write(message)
 
    if not st.session_state["chat_history"]:
        st.markdown(
            """
            Try:
            - Why is this model recommended?
            - What does the threshold mean?
            - When is calling everyone a reasonable baseline?
            - What factors drive the prediction?
            """
        )
        

def dashboard_screen() -> None:
    hide_loading_artifacts_on_dashboard()

    if (
        "cost_fp" not in st.session_state
        or "cost_fn" not in st.session_state
        or "dashboard_results" not in st.session_state
        or "dashboard_best" not in st.session_state
    ):
        set_page("home")
        st.rerun()
        st.stop()

    st.markdown('<div class="dashboard-header">', unsafe_allow_html=True)
    top_left, top_right = st.columns([0.8, 0.2])
    with top_left:
        st.title("Campaign Decision Dashboard")
        st.markdown(
            '<div class="section-note dashboard-subtitle">Model ranking, savings, sensitivity, explanations, and customer-level decisions</div>',
            unsafe_allow_html=True,
        )
    with top_right:
        st.write("")
        if st.button("Back / Home", width="stretch"):
            st.session_state.pop("dashboard_future", None)
            st.session_state.pop("dashboard_future_key", None)
            set_page("home")
            st.rerun()
            st.stop()
    st.markdown("</div>", unsafe_allow_html=True)

    cost_fp = float(st.session_state["cost_fp"])
    cost_fn = float(st.session_state["cost_fn"])

    cost_key = (cost_fp, cost_fn)
    if st.session_state.get("dashboard_cost_key") != cost_key:
        set_page("home")
        st.rerun()
        st.stop()

    bundle = st.session_state.get("dashboard_bundle")
    if bundle is None:
        bundle = cached_bundle()
        st.session_state["dashboard_bundle"] = bundle
    results = st.session_state["dashboard_results"]
    best = st.session_state.get("dashboard_best")

    if best is None:
        st.error("No models trained successfully. Check the dependency setup and training errors.")
        st.dataframe(results, width='stretch', hide_index=True)
        return

    tabs = st.tabs(
        [
            "Overview",
            "Model Comparison & Revenue-to-Cost Ratio Sensitivity",
            "Explainability",
            "Single Customer Decision",
            "AI Chatbot",
        ]
    )
    with tabs[0]:
        overview_tab(results, best, cost_fp, cost_fn)
    with tabs[1]:
        comparison_sensitivity_tab(bundle, results, cost_fp, cost_fn)
    with tabs[2]:
        explainability_tab(bundle, best)
    with tabs[3]:
        customer_input_tab(bundle, results, best, cost_fp, cost_fn)
    with tabs[4]:
        chatbot_tab(best, cost_fp, cost_fn)


def main() -> None:
    inject_style()
    page = st.session_state.get("app_page")
    if page is None:
        page = "dashboard" if st.session_state.get("dashboard_ready") else "home"
        st.session_state["app_page"] = page

    if page == "loading":
        loading_home_screen()
    elif page == "dashboard":
        dashboard_screen()
    else:
        home_screen()


if __name__ == "__main__":
    main()