"""
Module 6: VeriLens AI - Streamlit Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
import os

# Make sure modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from modules.input_handler import handle_text_input, handle_image_input, validate_payload
from modules.ocr_extractor import extract_text_from_image, clean_ocr_output
from modules.nlp_pipeline import preprocess_for_model, get_text_stats
from modules.model import predict
from modules.trust_score import calculate_trust_score

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VeriLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0e1a; color: #e8eaf0; }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a2035, #1e2d45);
        border: 1px solid #2a3f5f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }

    /* Trust score ring */
    .trust-ring {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Verdict badge */
    .verdict-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 8px 0;
    }

    /* Reason items */
    .reason-item {
        background: #111827;
        border-left: 3px solid #7c3aed;
        padding: 8px 14px;
        margin: 4px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.92rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #7c3aed;
        border-bottom: 1px solid #2a3f5f;
        padding-bottom: 6px;
        margin: 16px 0 10px 0;
    }

    /* Recommendation box */
    .rec-box {
        background: #0f1f12;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.95rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def verdict_color(verdict: str) -> str:
    return {
        "Likely True": "#22c55e",
        "Uncertain": "#f59e0b",
        "Likely Misleading": "#f97316",
        "Likely Fake": "#ef4444",
    }.get(verdict, "#6b7280")


def trust_gauge(score: int) -> go.Figure:
    color = (
        "#22c55e" if score >= 75 else
        "#f59e0b" if score >= 55 else
        "#f97316" if score >= 35 else
        "#ef4444"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"color": color, "size": 48}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
            "bar": {"color": color},
            "bgcolor": "#1a2035",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#3f1a1a"},
                {"range": [35, 55], "color": "#3f2a0a"},
                {"range": [55, 75], "color": "#2a2f0a"},
                {"range": [75, 100], "color": "#0a2f1a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e8eaf0"},
        margin={"t": 10, "b": 10, "l": 20, "r": 20},
    )
    return fig


def sub_score_bar_chart(sub_scores: dict) -> go.Figure:
    labels = list(sub_scores.keys())
    values = list(sub_scores.values())
    colors = ["#22c55e" if v >= 60 else "#f59e0b" if v >= 40 else "#ef4444" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont={"color": "#e8eaf0"},
    ))
    fig.update_layout(
        xaxis={"range": [0, 110], "showgrid": False, "color": "#6b7280"},
        yaxis={"color": "#e8eaf0"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin={"t": 10, "b": 10, "l": 10, "r": 40},
        showlegend=False,
    )
    return fig


def run_analysis(text: str):
    """Full pipeline: preprocess → model → trust score."""
    with st.spinner("🧠 Analyzing content..."):
        clean_text = preprocess_for_model(text)
        prediction = predict(clean_text)
        result = calculate_trust_score(text, prediction)
        stats = get_text_stats(text)
    return result, stats, prediction


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search.png", width=60)
    st.markdown("## VeriLens AI")
    st.markdown("*Explainable Trust Verification*")
    st.divider()

    st.markdown("### ℹ️ How It Works")
    st.markdown("""
1. **Paste** news text or **upload** a screenshot
2. Our AI model analyzes the content
3. Get an **Explainable Trust Score** with reasons
4. Decide whether to share or verify further
    """)

    st.divider()
    st.markdown("### 📊 Trust Score Legend")
    st.markdown("🟢 **75–100** — Likely True")
    st.markdown("🟡 **55–74** — Uncertain")
    st.markdown("🟠 **35–54** — Likely Misleading")
    st.markdown("🔴 **0–34** — Likely Fake")

    st.divider()
    st.caption("IEEE DataPort Hackathon 2026")
    st.caption("Built with DistilBERT + Explainability Layer")


# ── Main Header ────────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='font-size:2.8rem; font-weight:800; color:#e8eaf0;'>
        🔍 VeriLens <span style='color:#7c3aed;'>AI</span>
    </h1>
    <p style='color:#9ca3af; font-size:1.1rem;'>
        Trustworthy Information Verification · Explainable AI
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ── Input Section ──────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["📝 Paste Text / Article", "🖼️ Upload Screenshot"])

extracted_text = None
source_info = ""

with tab1:
    st.markdown("#### Enter news content to verify")
    user_text = st.text_area(
        label="News content",
        placeholder="Paste a news headline, article, or any piece of information here...",
        height=180,
        label_visibility="collapsed",
    )
    col_a, col_b = st.columns([1, 5])
    with col_a:
        analyze_text_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_text_btn and user_text.strip():
        extracted_text = user_text.strip()
        source_info = "Direct text input"

with tab2:
    st.markdown("#### Upload a screenshot of news content")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        col_img, col_ocr = st.columns([1, 1])
        with col_img:
            st.image(uploaded_file, caption="Uploaded Screenshot", use_column_width=True)
        with col_ocr:
            if st.button("🔍 Extract & Analyze", type="primary"):
                with st.spinner("📖 Extracting text via OCR..."):
                    payload = handle_image_input(uploaded_file.read(), uploaded_file.name)
                    raw_ocr = extract_text_from_image(payload.image)
                    extracted_text = clean_ocr_output(raw_ocr)
                    source_info = f"OCR from: {uploaded_file.name}"

                if extracted_text:
                    st.success(f"✅ Extracted {len(extracted_text.split())} words")
                    with st.expander("👁️ View extracted text"):
                        st.text(extracted_text)
                else:
                    st.error("❌ Could not extract text from image. Try a clearer screenshot.")

# ── Demo mode for quick testing ────────────────────────────────────────────────

with st.expander("🧪 Try a demo example"):
    demo_options = {
        "Fake News Example": (
            "BREAKING!!! SHOCKING: Scientists DISCOVER that drinking bleach CURES cancer!! "
            "You won't believe what doctors don't want you to know! Share before they DELETE this!!! "
            "Experts are furious. This one weird trick will change everything!!"
        ),
        "Real News Example": (
            "The Reserve Bank of India announced a 25 basis point reduction in the repo rate "
            "during its Monetary Policy Committee meeting on Wednesday. According to the official "
            "statement, the decision was taken unanimously to support economic growth. "
            "Data from the Ministry of Finance shows GDP growth at 6.8% for the last quarter."
        ),
        "Uncertain Example": (
            "Sources close to the government reportedly claim that a major policy change is "
            "imminent. While officials have not confirmed this, several insiders suggest the "
            "announcement could come this week. It is believed that the decision has already "
            "been finalized, though some say it might be delayed."
        ),
    }
    demo_choice = st.selectbox("Select a demo:", list(demo_options.keys()))
    if st.button("▶️ Run Demo"):
        extracted_text = demo_options[demo_choice]
        source_info = f"Demo: {demo_choice}"
        st.info(f"Using demo text: *{demo_choice}*")


# ── Results Section ────────────────────────────────────────────────────────────

if extracted_text:
    st.divider()
    st.markdown("## 📊 Analysis Results")

    result, stats, prediction = run_analysis(extracted_text)

    # ── Row 1: Trust Score + Verdict ─────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1.5, 1.3])

    with col1:
        st.markdown("#### Trust Score")
        st.plotly_chart(trust_gauge(result.trust_score), use_container_width=True)

    with col2:
        st.markdown("#### Verdict")
        color = verdict_color(result.verdict)
        st.markdown(f"""
        <div style='text-align:center; padding: 30px 20px;'>
            <div style='font-size:4rem;'>{result.verdict_emoji}</div>
            <div style='
                font-size:1.6rem;
                font-weight:800;
                color:{color};
                margin: 10px 0;
            '>{result.verdict}</div>
            <div style='color:#9ca3af; font-size:0.9rem;'>
                Model: {result.fake_prob:.0%} Fake · {result.real_prob:.0%} Real
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='rec-box'>
            <b>Recommendation:</b><br>{result.recommendation}
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("#### Text Stats")
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color:#9ca3af; font-size:0.8rem;'>WORDS</div>
            <div style='font-size:2rem; font-weight:700;'>{stats['word_count']}</div>
        </div>
        <div class='metric-card'>
            <div style='color:#9ca3af; font-size:0.8rem;'>SENTENCES</div>
            <div style='font-size:2rem; font-weight:700;'>{stats['sentence_count']}</div>
        </div>
        <div class='metric-card'>
            <div style='color:#9ca3af; font-size:0.8rem;'>AVG SENT. LENGTH</div>
            <div style='font-size:2rem; font-weight:700;'>{stats['avg_sentence_length']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Row 2: Reasons + Sub-scores ──────────────────────────────
    col4, col5 = st.columns([1.1, 0.9])

    with col4:
        st.markdown("#### 🔎 Why this score?")
        for reason in result.reasons:
            st.markdown(f"<div class='reason-item'>{reason}</div>", unsafe_allow_html=True)

    with col5:
        st.markdown("#### 📈 Component Breakdown")
        st.plotly_chart(
            sub_score_bar_chart(result.sub_scores),
            use_container_width=True,
        )

    st.divider()

    # ── Row 3: Analyzed text preview ─────────────────────────────
    with st.expander("📄 View analyzed content"):
        st.markdown(f"**Source:** {source_info}")
        st.text_area("Content", value=extracted_text, height=150, disabled=True, label_visibility="collapsed")

    # ── History tracking ──────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Snippet": extracted_text[:60] + "...",
        "Trust Score": result.trust_score,
        "Verdict": result.verdict,
        "Fake Prob": f"{result.fake_prob:.0%}",
    })

    if len(st.session_state.history) > 1:
        st.divider()
        st.markdown("#### 📋 Analysis History (this session)")
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

elif not any([
    (tab1 and user_text.strip()),
]):
    # Placeholder when nothing is entered
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color:#4b5563;'>
        <div style='font-size:4rem;'>🔍</div>
        <h3>Enter or upload news content above to begin verification</h3>
        <p>Supports text paste, article snippets, or screenshot images</p>
    </div>
    """, unsafe_allow_html=True)
