from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.ticker import PercentFormatter
import shap


PALETTE = {
    "ink": "#102033",
    "muted": "#53657a",
    "line": "#cfd8e3",
    "paper": "#eef4f8",
    "panel": "#ffffff",
    "navy": "#102033",
    "navy_2": "#1d3550",
    "teal": "#0f766e",
    "blue": "#2563a9",
    "sky": "#4f8fbf",
    "amber": "#b7791f",
    "coral": "#bd5b45",
    "violet": "#6356a6",
    "green": "#1b7f55",
    "rose": "#a9445d",
    "slate": "#64748b",
}

MODEL_COLORS = [
    PALETTE["teal"],
    PALETTE["blue"],
    PALETTE["navy_2"],
    PALETTE["amber"],
    PALETTE["coral"],
    PALETTE["violet"],
    PALETTE["green"],
]


def style_plot(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["ink"], size=13),
        title=dict(font=dict(size=18, color=PALETTE["ink"]), x=0.02, xanchor="left"),
        margin=dict(l=28, r=28, t=78, b=58),
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_color=PALETTE["ink"]),
        legend=dict(title_font=dict(color=PALETTE["ink"]), font=dict(color=PALETTE["ink"])),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color=PALETTE["ink"]),
        title_font=dict(color=PALETTE["ink"]),
    )
    fig.update_yaxes(
        gridcolor="#dde6ee",
        zeroline=False,
        tickfont=dict(color=PALETTE["ink"]),
        title_font=dict(color=PALETTE["ink"]),
    )
    return fig


def expected_profit_by_model(results: pd.DataFrame) -> go.Figure:
    valid = results[results["status"].eq("ok")].sort_values("expected_profit", ascending=False)
    fig = px.bar(
        valid,
        x="model",
        y="expected_profit",
        color="model",
        text="expected_profit",
        title="Expected Profit by Model ($)",
        color_discrete_sequence=MODEL_COLORS,
    )
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Expected profit ($)")
    max_profit = valid["expected_profit"].max() if not valid.empty else 0
    min_profit = valid["expected_profit"].min() if not valid.empty else 0
    fig.update_yaxes(
        tickprefix="$",
        range=[min(min_profit * 1.2, 0), max_profit * 1.2 if max_profit > 0 else 1],
    )
    return style_plot(fig)


def confusion_matrix_figure(best: dict) -> go.Figure:
    cm = np.array([[best["tn"], best["fp"]], [best["fn"], best["tp"]]])
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["No", "Yes"],
        y=["No", "Yes"],
        color_continuous_scale=["#edf6f5", "#8dc8c1", PALETTE["teal"], PALETTE["navy"]],
        title=f"Confusion Matrix: {best['model']}",
    )
    fig.update_layout(coloraxis_showscale=False)
    return style_plot(fig)


def focused_model_comparison(focused: pd.DataFrame) -> go.Figure:
    source = focused.copy()
    source["display_model"] = source["model"].replace({"Call Everyone": "Call Everyone Baseline"})
    source["comparison_label"] = source.apply(
        lambda row: (
            f"vs Call all ${row['profit_gain_vs_call_everyone']:,.0f}<br>"
            f"vs Logistic ${row['profit_gain_vs_logistic']:,.0f}<br>"
            f"F-beta {row['f_beta']:.3f}"
        ),
        axis=1,
    )
    colors = [
        PALETTE["teal"] if idx == 0 else PALETTE["blue"] if row["status"] == "ok" else PALETTE["amber"]
        for idx, row in source.reset_index(drop=True).iterrows()
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=source["display_model"],
            y=source["expected_profit"],
            name="Expected profit",
            marker_color=colors,
            text=source["comparison_label"],
            textposition="outside",
            customdata=np.stack(
                [
                    source["f_beta"].fillna(0),
                    source["profit_gain_vs_logistic"].fillna(0),
                    source["profit_gain_vs_call_everyone"].fillna(0),
                    source["precision"].fillna(0),
                    source["recall"].fillna(0),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Expected profit: $%{y:,.0f}<br>"
                "Profit gain vs logistic: $%{customdata[1]:,.0f}<br>"
                "Profit gain vs call everyone: $%{customdata[2]:,.0f}<br>"
                "F-beta: %{customdata[0]:.3f}<br>"
                "Precision: %{customdata[3]:.1%}<br>"
                "Recall: %{customdata[4]:.1%}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Focused Decision Comparison: Expected Profit, Profit Gains and F-beta",
        xaxis_title="",
        yaxis_title="Expected profit ($)",
        showlegend=False,
        bargap=0.34,
    )
    max_profit = source["expected_profit"].max() if not source.empty else 0
    min_profit = source["expected_profit"].min() if not source.empty else 0
    fig.update_yaxes(
        tickprefix="$",
        range=[min(min_profit * 1.35, 0), max_profit * 1.35 if max_profit > 0 else 1],
    )
    return style_plot(fig)


def model_ratio_curves(sensitivity_detail: pd.DataFrame, winner_models: list[str] | None = None) -> go.Figure:
    source = sensitivity_detail.copy()
    source = source[source["profit_gain_vs_call_everyone"] > 0]

    fig = px.line(
        source,
        x="cost_ratio",
        y="profit_gain_vs_call_everyone",
        color="model",
        markers=True,
        title="Models Net Profit Against Call-Everyone Baseline Across Revenue-to-Cost Ratios ($)",
        labels={
            "profit_gain_vs_call_everyone": "Profit vs Call Everyone Baseline ($)",
            "cost_ratio": "Revenue-to-Cost Ratio ($ gain per call / $ cost per call)",
            "model": "Model",
        },
        color_discrete_sequence=MODEL_COLORS,
        custom_data=["expected_profit"],
    )
    fig.update_traces(line=dict(width=4), marker=dict(size=9, line=dict(width=1.5, color="#ffffff")))
    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Revenue-to-Cost Ratio: %{x:.2f}<br>"
            "Profit Gap: $%{y:,.0f}<br>"
            "Expected Profit: $%{customdata[0]:,.0f}<extra></extra>"
        )
    )
    fig.update_yaxes(tickprefix="$")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
    return style_plot(fig)


def _format_probability_text(text: str) -> str:
    try:
        value = float(text.replace("−", "-"))
    except ValueError:
        return text
    return f"{value:+.1%}".replace("-", "−")


def render_shap_summary_plot(explanation, transformed: pd.DataFrame, model_name: str, st, max_display: int = 20) -> None:
 
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, -1]
    plt.close("all")
    shap.summary_plot(
        values,
        transformed,
        feature_names=list(transformed.columns),
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Summary Plot - {model_name}", fontsize=14, fontweight="bold", loc="center")
    plt.tight_layout()
    st.pyplot(plt.gcf(), width='stretch', clear_figure=True)
 
 
def render_shap_dependence_plot(
    explanation,
    transformed: pd.DataFrame,
    feature_name: str,
    model_name: str,
    st,
    raw_df: pd.DataFrame | None = None,
) -> None:
    """Render a SHAP dependence plot.
 
    If raw_df is provided and contains a column matching feature_name (after
    stripping the ColumnTransformer's prefix, e.g. "robust__age" -> "age"),
    the x-axis values are swapped for the original raw-scale values so the
    plot reads in real units (years, rates, etc.) instead of scaled values.
    """
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, -1]
    plt.close("all")
    shap.dependence_plot(
        feature_name,
        values,
        transformed,
        feature_names=list(transformed.columns),
        show=False,
    )
 
    # Relabel the x-axis with raw-scale values for numeric features that were
    # scaled by the preprocessor, so the plot is human-readable.
    raw_col = feature_name.split("__")[-1]
    if raw_df is not None and raw_col in raw_df.columns and pd.api.types.is_numeric_dtype(raw_df[raw_col]):
        fig = plt.gcf()
        ax = fig.axes[0] if fig.axes else plt.gca()
        scaled_values = transformed[feature_name].to_numpy()
        raw_values = raw_df[raw_col].to_numpy()
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if offsets.shape[0] != len(scaled_values):
                continue
            new_offsets = offsets.copy()
            # Map each plotted scaled x-value to its corresponding raw value
            # by matching on the scaled value (dependence_plot uses the
            # transformed column directly as the x-coordinate).
            order = np.argsort(scaled_values)
            new_offsets[order, 0] = raw_values[order]
            collection.set_offsets(new_offsets)
        ax.set_xlim(raw_values.min(), raw_values.max())
        ax.set_xlabel(raw_col)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=False)
 
    plt.title(f"SHAP Dependence Plot - {model_name}", fontsize=14, fontweight="bold", loc="center")
    plt.tight_layout()
    st.pyplot(plt.gcf(), width='stretch', clear_figure=True)
 
 
def render_probability_waterfall_plot(
    single,
    probability: float,
    model_name: str,
    st,
    max_display: int = 8,
    raw_row: pd.Series | None = None,
) -> None:
    """Render a SHAP probability-space waterfall plot for one customer.
 
    If raw_row is provided (a single row of raw-scale feature values, indexed
    by the original column names such as "age", "euribor3m"), the per-feature
    "feature = value" labels are rewritten using those raw values instead of
    the scaled/encoded values the model actually saw, so the plot is
    human-readable.
    """
    plt.close("all")
    plt.figure(figsize=(20, 12))
 
    if raw_row is not None:
        single = copy.copy(single)
        new_data = np.asarray(single.data, dtype=object).copy()
        for i, transformed_name in enumerate(single.feature_names):
            raw_col = transformed_name.split("__")[-1]
            if raw_col in raw_row.index:
                new_data[i] = raw_row[raw_col]
        single.data = new_data
 
    shap.plots.waterfall(single, max_display=max_display, show=False)
    fig = plt.gcf()
    expected_value = float(np.asarray(single.base_values).reshape(-1)[0])
    for ax in fig.axes:
        ax.margins(x=0.15)
    main_ax = fig.axes[0] if fig.axes else plt.gca()
    main_ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    for text in main_ax.texts:
        text.set_text(_format_probability_text(text.get_text()))
        text.set_fontsize(11)
    fig.text(
        0.5,
        0.93,
        f"Baseline Probability = {expected_value:.1%} | Predicted Probability = {probability:.1%}",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="semibold",
        color="dimgray",
    )
    fig.suptitle(f"SHAP Waterfall Plot - {model_name}", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    st.pyplot(fig, width='stretch', clear_figure=True)