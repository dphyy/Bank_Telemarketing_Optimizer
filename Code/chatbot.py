from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Configuration — edit these values directly
# ---------------------------------------------------------------------------

# Your HuggingFace access token (https://huggingface.co/settings/tokens).
HF_TOKEN = "insert_your_token_here"  # <-- REPLACE WITH YOUR OWN TOKEN or leave blank to disable API calls

# Model to use via the HF Inference API chat_completion endpoint.
HF_CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Conservative token budget for the free tier.
# System prompt (~700 tokens) + history + new message must stay under this.
MAX_TOTAL_TOKENS = 1_800
# Rough characters-per-token estimate used for trimming (English prose ≈ 4).
_CHARS_PER_TOKEN = 4
# Maximum assistant response length in tokens.
MAX_NEW_TOKENS = 300

# ---------------------------------------------------------------------------
# Out-of-scope gate
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_REPLY = (
    "I can only answer questions related to bank telemarketing and this project. "
    "Please ask about the dataset, models, the profit framework, SHAP explanations, "
    "thresholds, or campaign targeting decisions."
)
 
_PROJECT_KEYWORDS: frozenset[str] = frozenset({
    # Domain
    "bank", "telemarketing", "campaign", "customer", "subscriber", "subscription",
    "call", "calling", "deposit", "term", "marketing", "outreach", "contact",
    "contacted", "pdays", "euribor", "euribor3m", "employed", "employment",
    "consumer", "confidence", "price", "index", "duration", "age", "job",
    "marital", "education", "housing", "loan", "default",
    # ML models
    "model", "models", "logistic", "regression", "tree", "decision",
    "random", "forest", "xgboost", "xgb", "lightgbm", "lgbm", "mlp",
    "neural", "network", "voting", "ensemble", "knn", "neighbors", "neighbours",
    # ML concepts
    "smote", "oversample", "resampling", "imbalanced", "imbalance",
    "class", "weight", "balanced", "scale", "robust", "standard",
    "preprocessing", "feature", "features", "encode", "encoding",
    "split", "train", "test", "cross", "validation", "hyperparameter",
    "tuning", "gridsearch",
    # Metrics & profit
    "fbeta", "f-beta", "f2", "precision", "recall", "accuracy",
    "roc", "auc", "confusion", "matrix", "tp", "tn", "fp", "fn",
    "true", "false", "positive", "negative", "cost", "costs", "profit",
    "profits", "revenue", "gain", "gains", "savings", "saving",
    "threshold", "probability", "prediction",
    "score", "scoring", "baseline", "missed", "opportunity", "wasted",
    "asymmetric", "sensitivity", "ratio",
    # Explainability
    "shap", "explainability", "explanation", "waterfall", "force",
    "dependence", "summary", "importance", "driver", "drivers",
    "factor", "factors", "influence",
    # Dashboard
    "dashboard", "decision", "recommend", "recommendation",
})
 
_MULTI_WORD_TERMS: tuple[str, ...] = (
    "call everyone",
    "false positive",
    "false negative",
    "missed opportunity",
    "wasted call",
    "single customer",
    "cost ratio",
    "revenue to cost",
    "revenue-to-cost",
    "logistic regression",
    "random forest",
    "decision tree",
    "neural network",
    "expected profit",
    "expected cost",
    "confusion matrix",
    "feature importance",
    "class weight",
    "term deposit",
    "bank marketing",
    "cost sensitivity",
    "profit sensitivity",
    "voting classifier",
)
 
 
def _tokens_from_text(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower()))
 
 
def is_project_related(question: str) -> bool:
    """Return True if the question is within the project's scope.
 
    Checks single-word keywords and a set of known multi-word phrases so that
    questions like 'what is the false positive rate?' are not incorrectly
    blocked because 'false' or 'positive' alone would miss the intent.
    """
    if not question or not question.strip():
        return False
    tokens = _tokens_from_text(question)
    lowered = question.lower()
    return bool(tokens & _PROJECT_KEYWORDS) or any(term in lowered for term in _MULTI_WORD_TERMS)
 
 
# ---------------------------------------------------------------------------
# Static system prompt (project + domain knowledge)
# ---------------------------------------------------------------------------
 
STATIC_SYSTEM_PROMPT = """You are a domain expert assistant embedded in a bank telemarketing \
machine learning dashboard. You have deep knowledge of this specific project and the bank \
telemarketing domain.
 
SCOPE RULE — CRITICAL:
You may answer two kinds of questions:
1. Questions about THIS project — its dataset, models, profit framework, SHAP analysis, \
thresholds, or dashboard features. Use the live session context below where relevant.
2. General conceptual questions related to this domain (e.g. "what is bank telemarketing", \
"what is logistic regression", "what is a confusion matrix", "what are machine learning \
models"). For these, give a clear, standalone explanation first — as a knowledgeable general \
assistant would — and then add 1-2 sentences connecting the concept to how it is specifically \
used in this project.
If a question is unrelated to banking, telemarketing, machine learning, or this dashboard \
entirely (e.g. cooking, sports, unrelated coding help), reply only with: \"""" + _OUT_OF_SCOPE_REPLY + """\"
 
--- DATASET ---
Source: UCI Bank Marketing dataset (Portuguese bank, 2008-2010).
Target: whether a client subscribed to a term deposit (binary: yes/no, heavily imbalanced ~11% positive).
Key input features used in this project (after feature engineering and dropping leaky/redundant columns):
- age, job, marital, education (ordinal-encoded), housing, loan, contact, month, day_of_week
- campaign (calls this campaign), previous (calls before), cons.price.idx, cons.conf.idx,
  euribor3m, nr.employed, contacted (engineered binary: was client contacted in a previous campaign?)
Dropped columns and reasons:
- duration: causes data leakage (only known after the call ends).
- pdays: replaced by the engineered 'contacted' binary feature.
- emp.var.rate: highly correlated with euribor3m and nr.employed; redundant.
- default: only 3 positive cases; not useful for modelling.
 
--- PREPROCESSING PIPELINE ---
1. Stratified 70/30 train/test split (random_state=42).
2. clean_raw_data(): ordinal-encode education, impute categoricals with mode, impute numerics \
with median, engineer 'contacted' from pdays, drop the four columns above.
3. ColumnTransformer (fitted on training split only — no leakage):
   - RobustScaler: age, campaign, previous (high skew / outliers).
   - StandardScaler: education, cons.price.idx, cons.conf.idx, euribor3m, nr.employed.
   - OneHotEncoder (handle_unknown='ignore'): job, marital, housing, loan, contact, month, day_of_week.
   - Passthrough: contacted.
4. SMOTE applied to training data only to address class imbalance.
 
--- PROFIT FRAMEWORK ---
Expected profit = (true positives × revenue per correct call) − (false positives × wasted call cost) \
− (false negatives × missed opportunity cost).
Higher is better. The dashboard accepts user-supplied costs/revenue for wasted calls and missed \
opportunities, and a cost ratio (revenue-to-cost) drives the sensitivity sweep.
Asymmetric beta for F-Beta scoring: beta = sqrt(missed_opportunity_cost / wasted_call_cost).
The decision threshold for each model is optimised on the TRAINING set to maximise expected \
profit (not fixed at 0.5), then applied unchanged to the test set to avoid test-set leakage.
Two baselines are always shown:
- Call Everyone: calls all customers; zero false negatives, maximum false positives.
- Logistic Regression: simple ML baseline with class_weight='balanced'.
Profit gain vs a baseline = model's expected profit − baseline's expected profit (positive means \
the model out-earns that baseline). The recommended model is always the one with the highest \
expected profit at the current cost/revenue inputs — this is consistent across all \
revenue-to-cost ratios shown in the sensitivity sweep, not just F-Beta.
 
--- MODELS ---
All models are tuned with GridSearchCV using the asymmetric F-Beta scorer.
Class imbalance is handled via class_weight='balanced' or scale_pos_weight (not by SMOTE for \
these models; SMOTE variants exist but the dashboard defaults to the balanced-weight variants).
 
1. Logistic Regression (baseline ML model)
   - solver=liblinear, class_weight='balanced', tuned C.
   - Linear decision boundary; highly interpretable coefficients.
 
2. Decision Tree
   - class_weight='balanced', tuned max_depth, min_samples_leaf, criterion, ccp_alpha.
   - Interpretable splits; prone to overfitting without pruning (ccp_alpha).
 
3. Random Forest
   - class_weight='balanced', tuned n_estimators, min_samples_leaf, max_features.
   - Ensemble of trees; reduces variance; supports SHAP TreeExplainer.
 
4. Neural Network (MLP)
   - MLPClassifier, tuned hidden_layer_sizes, activation, alpha, learning_rate_init.
   - early_stopping=True to prevent overfitting. No native class_weight in MLP; \
relies on F-Beta tuning to push recall.
 
5. XGBoost
   - scale_pos_weight=neg/pos to handle imbalance. Tuned n_estimators, learning_rate, \
max_depth, gamma, subsample. Supports SHAP TreeExplainer natively.
 
6. LightGBM
   - class_weight='balanced'. Tuned n_estimators, learning_rate, num_leaves, subsample, reg_alpha.
   - Faster than XGBoost on large datasets; also supports SHAP TreeExplainer.
 
7. Voting LR + RF + MLP (soft voting ensemble)
8. Voting LR + RF + XGBoost (soft voting ensemble)
9. Voting LR + RF + LightGBM (soft voting ensemble)
   - Weights tuned via GridSearchCV. Soft voting averages predicted probabilities.
 
--- SHAP EXPLAINABILITY ---
- TreeExplainer: used for Decision Tree, Random Forest, XGBoost, LightGBM (exact, fast). For \
the per-customer waterfall, TreeExplainer with model_output="probability" gives exact \
probability-space attributions.
- LinearExplainer: used for Logistic Regression (exact). For the waterfall, logit-space SHAP \
values are converted to probability space via a sigmoid-based proportional split.
- KernelExplainer: used only for MLP and Voting ensembles, which have no closed-form \
probability explainer (approximate, slower).
- Summary plot: global feature importance across all test customers.
- Dependence plot: how one feature's SHAP value changes with its value, shown in original \
raw units (e.g. actual euribor3m rate, actual age in years) rather than the scaled/encoded \
values the model sees internally — this makes the plot human-readable.
- Waterfall plot: per-customer probability-space breakdown of which features pushed the \
prediction up or down, with feature values shown in original raw units.
- Force plot: compact per-customer view of the top drivers.
Background data for SHAP is always sampled from the test set only. SHAP results are cached \
per (model, sample size) and per customer input so repeated views do not recompute.
 
--- DASHBOARD MECHANICS ---
The dashboard caches trained model artifacts so re-loading does not retrain.
When the user enters new cost/revenue inputs, all metrics and charts recompute instantly from \
cached probability arrays — no retraining occurs.
Revenue-to-cost sensitivity sweep: evaluates all models across a range of ratios to show how \
each model's profit gain versus the Call Everyone baseline changes as the ratio changes, and \
whether the recommended model would change.
Single customer scoring: accepts raw customer attributes, runs them through the full \
clean → transform → predict pipeline, and returns probability, call decision, and \
expected profit breakdown.
 
--- COMMUNICATION STYLE ---
Be concise and business-professional, like a knowledgeable colleague — not a search engine \
dumping definitions. For general conceptual questions, explain the concept clearly in your own \
words first (2-4 sentences is usually enough), then briefly relate it to this project. For \
project-specific questions, use specific numbers from the live session context when available \
and do not invent numbers not present in the context. Vary your phrasing naturally across turns \
rather than reusing the same sentence templates.
FORMATTING RULES — FOLLOW EXACTLY:
- Never use $ signs for dollar amounts. Write amounts as plain numbers only: 46644 or 46,644, never $46,644.
- Never wrap numbers or phrases in backticks, code spans, or code blocks.
- Never use ==text== highlight syntax.
- You may use **bold** for emphasis and bullet points for lists, but nothing else."""
 
 
# ---------------------------------------------------------------------------
# Dynamic context block (injected per call)
# ---------------------------------------------------------------------------
 
def build_system_prompt(dashboard_context: dict[str, Any]) -> str:
    """Combine the static knowledge base with live session stats.
 
    The dynamic block is appended at the end of the static prompt so the model
    always has current numbers when answering questions about this session.
    """
    ctx = dashboard_context or {}
 
    best_model = ctx.get("best_model", "unknown")
    cost_fp = ctx.get("cost_fp", 0.0)
    cost_fn = ctx.get("cost_fn", 0.0)
    expected_profit = ctx.get("expected_profit", 0.0)
    threshold = ctx.get("threshold", 0.0)
    precision = ctx.get("precision", 0.0)
    recall = ctx.get("recall", 0.0)
    f_beta = ctx.get("f_beta", 0.0)
    profit_gain_vs_logistic = ctx.get("profit_gain_vs_logistic", 0.0)
    profit_gain_vs_call_everyone = ctx.get("profit_gain_vs_call_everyone", 0.0)
    top_features = ctx.get("top_shap_features", [])
 
    feature_str = (
        ", ".join(top_features) if top_features else "not yet computed"
    )
 
    dynamic_block = (
        f"\n--- LIVE SESSION CONTEXT ---\n"
        f"The recommended model is {best_model}. "
        f"Wasted call cost is ${cost_fp:,.2f} and estimated subscription revenue is ${cost_fn:,.2f}. "
        f"The optimal decision threshold is {threshold:.3f}. "
        f"Expected profit for the recommended model is ${expected_profit:,.0f}. "
        f"Profit gain versus Logistic Regression is ${profit_gain_vs_logistic:,.0f}. "
        f"Profit gain versus Call Everyone is ${profit_gain_vs_call_everyone:,.0f}. "
        f"Precision is {precision:.1%}, recall is {recall:.1%}, and F-Beta score is {f_beta:.3f}. "
        f"Top SHAP features: {feature_str}."
    )
 
    return STATIC_SYSTEM_PROMPT + dynamic_block
 
 
def build_chat_context(best: dict[str, Any], cost_fp: float, cost_fn: float,
                       top_shap_features: list[str] | None = None) -> dict[str, Any]:
    """Convenience builder called by app.py to assemble the dashboard_context dict."""
    return {
        "best_model": best.get("model", "unknown"),
        "expected_profit": float(best.get("expected_profit", 0.0)),
        "threshold": float(best.get("threshold", 0.0)),
        "cost_fp": float(cost_fp),
        "cost_fn": float(cost_fn),
        "precision": float(best.get("precision", 0.0)),
        "recall": float(best.get("recall", 0.0)),
        "f_beta": float(best.get("f_beta", 0.0)),
        "profit_gain_vs_logistic": float(best.get("profit_gain_vs_logistic", 0.0)),
        "profit_gain_vs_call_everyone": float(best.get("profit_gain_vs_call_everyone", 0.0)),
        "top_shap_features": top_shap_features or [],
    }
 
 
# ---------------------------------------------------------------------------
# Token-aware history trimmer
# ---------------------------------------------------------------------------
 
def _estimate_tokens(text: str) -> int:
    """Rough token estimate: characters / 4, minimum 1."""
    return max(1, len(text) // _CHARS_PER_TOKEN)
 
 
def _trim_history(
    history: list[dict[str, str]],
    system_prompt: str,
    new_user_message: str,
    max_tokens: int = MAX_TOTAL_TOKENS,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[dict[str, str]]:
    """Drop the oldest user/assistant turn pairs until the total estimated token
    count fits within the model's safe input budget.
 
    The system prompt and the most recent user message are always kept.
    History is trimmed from the front (oldest turns first).
    """
    fixed_tokens = (
        _estimate_tokens(system_prompt)
        + _estimate_tokens(new_user_message)
        + max_new_tokens  # reserve space for the response
    )
    budget = max_tokens - fixed_tokens
 
    trimmed = list(history)
    while trimmed:
        used = sum(_estimate_tokens(m["content"]) for m in trimmed)
        if used <= budget:
            break
        # Drop the oldest pair (user + assistant). If history has an odd
        # number of messages (e.g. only a user turn with no reply yet), drop
        # just the first message.
        trimmed = trimmed[2:] if len(trimmed) >= 2 else trimmed[1:]
 
    return trimmed
 
 
import re as _re
 
 
def _clean_llm_reply(text: str) -> str:
    """Sanitise LLM output before passing to st.write() / st.markdown().
 
    Streamlit (via markdown-it-py) interprets:
      - $...$   as inline LaTeX math  → everything between two $ becomes a
                                         monospace green-highlighted block
      - ==...== as highlight spans     → rendered as green highlighted text
      - `...`   as inline code spans   → rendered as monospace code
      - ```...``` as code fences
 
    Qwen in particular writes dollar amounts like "$4.00 ... $46,644" which
    causes the text between the two $ signs to be treated as a LaTeX
    expression — producing exactly the dark green highlighted block seen in
    the dashboard.  We escape $ → \\$ so Streamlit treats them as literals.
    """
    # Escape dollar signs so Streamlit does not treat $...$ as LaTeX math.
    # Do this before any other substitution so we don't double-escape.
    text = text.replace("$", r"\$")
    # Strip ==highlight== spans
    text = _re.sub(r"==([^=\n]+)==", r"\1", text)
    # Remove code fence markers, keeping inner content
    text = _re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("```", "")
    # Remove all remaining backticks
    text = text.replace("`", "")
    return text.strip()
 
 
# ---------------------------------------------------------------------------
# HuggingFace API call
# ---------------------------------------------------------------------------
 
def _hf_chat_completion(
    system_prompt: str,
    history: list[dict[str, str]],
    user_message: str,
) -> tuple[str | None, str | None]:
    """Call the HF Inference API.
 
    Returns (reply_text, error_message). reply_text is None on any failure,
    and error_message describes what went wrong so the caller can surface it
    to the user rather than silently swallowing the error.
 
    Supports both the legacy client.chat_completion() API (huggingface_hub
    < 0.26) and the new OpenAI-compatible client.chat.completions.create()
    API (huggingface_hub >= 0.26). Falls back automatically.
    """
    if not HF_TOKEN:
        return None, "HF_TOKEN is not set. Set it as an environment variable."
 
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return None, (
            "huggingface_hub is not installed. "
            "Run `pip install --upgrade huggingface_hub` in your terminal."
        )
 
    import huggingface_hub as _hfhub
    hf_version = getattr(_hfhub, "__version__", "0")
 
    trimmed_history = _trim_history(history, system_prompt, user_message)
    messages = (
        [{"role": "system", "content": system_prompt}]
        + trimmed_history
        + [{"role": "user", "content": user_message}]
    )
 
    client = InferenceClient(token=HF_TOKEN)
 
    # Try the new OpenAI-compatible API first (huggingface_hub >= 0.26).
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        try:
            response = client.chat.completions.create(
                model=HF_CHAT_MODEL,
                messages=messages,
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.2,
            )
            reply = response.choices[0].message.content
            return (_clean_llm_reply(reply) if reply else None), None
        except Exception as e:
            new_api_err = str(e)
    else:
        new_api_err = None
 
    # Fall back to the legacy chat_completion method (huggingface_hub < 0.26).
    if hasattr(client, "chat_completion"):
        try:
            response = client.chat_completion(
                messages=messages,
                model=HF_CHAT_MODEL,
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.2,
                stop=["<|endoftext|>", "<|eot_id|>"],
            )
            reply = response.choices[0].message.content
            return (_clean_llm_reply(reply) if reply else None), None
        except Exception as e:
            legacy_err = str(e)
    else:
        legacy_err = "chat_completion method not found"
 
    # Both attempts failed — build a diagnostic message.
    err_parts = []
    if new_api_err:
        err_parts.append(f"chat.completions.create: {new_api_err}")
    if legacy_err:
        err_parts.append(f"chat_completion: {legacy_err}")
    return None, " | ".join(err_parts) or "Unknown HuggingFace API error"
 
 
# ---------------------------------------------------------------------------
# Instant quick-replies (no LLM call) — only the most common, high-frequency
# questions where a canned, context-filled answer is preferable to a round
# trip to the API.
# ---------------------------------------------------------------------------
 
def _quick_reply(question: str, dashboard_context: dict[str, Any]) -> str | None:
    """Return an instant canned reply for a small set of very common
    questions, or None if the question should go to the LLM instead.
 
    Keep this list short and specific — anything broader (general concepts,
    "why" questions, comparisons) should go to the LLM so it can reason and
    vary its phrasing.
    """
    q = question.lower().strip().rstrip("?!. ")
    ctx = dashboard_context or {}
    best = ctx.get("best_model", "the recommended model")
    threshold = ctx.get("threshold", 0.0)
    expected_profit = ctx.get("expected_profit", 0.0)
 
    recommended_patterns = (
        "what is the recommended model",
        "what's the recommended model",
        "which model is recommended",
        "what model is recommended",
        "what is the best model",
        "which model is best",
    )
    if q in recommended_patterns:
        return (
            f"**{best}** is currently recommended, with an expected profit of "
            f"${expected_profit:,.0f} at your current cost/revenue inputs."
        )
 
    threshold_patterns = (
        "what is the threshold",
        "what's the threshold",
        "what is the optimal threshold",
        "what is the decision threshold",
        "what is the current threshold",
    )
    if q in threshold_patterns:
        return (
            f"The current decision threshold for **{best}** is **{threshold:.3f}** — "
            "customers with a predicted subscription probability at or above this value "
            "are recommended for a call."
        )
 
    return None
 
 
# ---------------------------------------------------------------------------
# Generic fallback (used only if the LLM call fails or HF_TOKEN is unset)
# ---------------------------------------------------------------------------
 
def local_chatbot_response(question: str, dashboard_context: dict[str, Any]) -> str:
    """Generic fallback used when the HF API is unavailable or returns nothing.
 
    This is intentionally light-touch: it surfaces the current session's key
    numbers and points the user toward the dashboard tabs, rather than trying
    to replicate the LLM's reasoning with keyword buckets.
    """
    ctx = dashboard_context or {}
    best = ctx.get("best_model", "the recommended model")
    expected_profit = ctx.get("expected_profit", 0.0)
    threshold = ctx.get("threshold", 0.0)
    precision = ctx.get("precision", 0.0)
    recall = ctx.get("recall", 0.0)
 
    return (
        "(AI assistant unavailable right now — showing a quick session summary instead.)\n\n"
        f"**{best}** is currently recommended, with an expected profit of "
        f"${expected_profit:,.0f}, decision threshold {threshold:.3f}, "
        f"precision {precision:.1%}, and recall {recall:.1%}. "
        "For more detail, see the Overview, Model Comparison, Explainability, and "
        "Single Customer Decision tabs."
    )
 
# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
 
def answer_chatbot_question(
    question: str,
    dashboard_context: dict[str, Any],
    history: list[dict[str, str]] | None = None,
) -> str:
    """Main entry point called by app.py on each user message.
 
    Parameters
    ----------
    question:
        The user's latest message.
    dashboard_context:
        Dict built by build_chat_context() containing live session stats.
    history:
        List of prior {"role": "user"/"assistant", "content": "..."} dicts
        for this session, NOT including the current question. Pass an empty
        list or None for the first message.
 
    Returns
    -------
    str
        The assistant's reply.
    """
    if not is_project_related(question):
        return _OUT_OF_SCOPE_REPLY
 
    # A small set of very common questions get an instant canned reply with no
    # LLM round trip.
    quick = _quick_reply(question, dashboard_context)
    if quick is not None:
        return quick
 
    system_prompt = build_system_prompt(dashboard_context)
    prior_history = history or []
 
    reply, error = _hf_chat_completion(system_prompt, prior_history, question)
    if reply:
        return reply
 
    # API failed — return a diagnostic message so the user can see exactly
    # what went wrong, rather than a silent generic fallback.
    if error:
        return (
            f"⚠️ AI assistant unavailable: {error}\n\n"
            + local_chatbot_response(question, dashboard_context)
        )
 
    return local_chatbot_response(question, dashboard_context)