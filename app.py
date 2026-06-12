"""
app.py — M-SHIELD Streamlit Frontend
"""

import streamlit as st
import tempfile, os, time
from mshield import mshield_analyze, load_models, TOOL_REGISTRY

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="M-SHIELD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #FAFAF9; }
    .block-container { padding: 2rem 2.5rem; max-width: 1100px; }
    h1 { font-size: 1.6rem; font-weight: 500; }
    h3 { font-size: 1rem; font-weight: 500; color: #444; }

    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 12px;
    }
    .badge-safe     { background:#EAF3DE; color:#3B6D11; }
    .badge-caution  { background:#E6F1FB; color:#185FA5; }
    .badge-warning  { background:#FAEEDA; color:#854F0B; }
    .badge-critical { background:#FAECE7; color:#993C1D; }

    .signal-box {
        background: #F4F2FB;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .tool-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 3px;
    }
    .tool-allowed { background:#EAF3DE; color:#3B6D11; }
    .tool-blocked { background:#FCEBEB; color:#A32D2D; text-decoration: line-through; }

    .flag-fired   { color: #D85A30; font-weight: 500; }
    .flag-clear   { color: #639922; }

    .stButton > button {
        background: #534AB7;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 14px;
        font-weight: 500;
        width: 100%;
    }
    .stButton > button:hover { background: #3C3489; }
</style>
""", unsafe_allow_html=True)

# ── Load models (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()

# ── Header ───────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown("""
    <div style="width:48px;height:48px;background:#534AB7;border-radius:10px;
         display:flex;align-items:center;justify-content:center;
         color:#EEEDFE;font-size:20px;font-weight:500;margin-top:4px">M</div>
    """, unsafe_allow_html=True)
with col_title:
    st.markdown("## M-SHIELD")
    st.caption("A Risk-Aware Multimodal Defense Framework Against Indirect Prompt Injection")

st.divider()

# ── Input section ────────────────────────────────────────────────
st.markdown("### Analyze input")
tab_img, tab_audio, tab_text = st.tabs(["🖼 Image", "🔊 Audio", "📝 Text"])

input_data  = None
input_label = None
modality    = None

with tab_img:
    uploaded_img = st.file_uploader(
        "Upload an image (PNG, JPG)",
        type=["png","jpg","jpeg"],
        key="img_upload"
    )
    if uploaded_img:
        st.image(uploaded_img, width=400)
        input_label = uploaded_img.name
        modality    = "image"

with tab_audio:
    uploaded_audio = st.file_uploader(
        "Upload an audio file (WAV, MP3)",
        type=["wav","mp3","flac"],
        key="audio_upload"
    )
    if uploaded_audio:
        st.audio(uploaded_audio)
        input_label = uploaded_audio.name
        modality    = "audio"

with tab_text:
    text_input = st.text_area(
        "Paste or type text content",
        placeholder="Enter document text, email content, or any text to analyze...",
        height=150,
        key="text_input"
    )
    if text_input.strip():
        input_label = "text_input"
        modality    = "text"

st.markdown("")
analyze_btn = st.button("Analyze with M-SHIELD")

st.divider()

# ── Analysis ─────────────────────────────────────────────────────
if analyze_btn:
    if modality is None:
        st.warning("Please provide an input — image, audio, or text.")
    else:
        with st.spinner("Loading M-SHIELD models (first run takes 2-3 minutes)..."):
            models = get_models()

        with st.spinner("Analyzing..."):
            # Save uploaded file to temp path if needed
            if modality == "image":
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp:
                    tmp.write(uploaded_img.getbuffer())
                    tmp_path = tmp.name
                result = mshield_analyze(tmp_path, input_label, models)
                os.unlink(tmp_path)

            elif modality == "audio":
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp.write(uploaded_audio.getbuffer())
                    tmp_path = tmp.name
                result = mshield_analyze(tmp_path, input_label, models)
                os.unlink(tmp_path)

            else:
                result = mshield_analyze(text_input, input_label, models)

        # ── Results ──────────────────────────────────────────────
        st.markdown("### Results")

        # Risk badge
        level = result['restriction_level']
        badge_class = {
            "SAFE"    : "badge-safe",
            "CAUTION" : "badge-caution",
            "WARNING" : "badge-warning",
            "CRITICAL": "badge-critical",
        }.get(level, "badge-caution")

        st.markdown(
            f'<span class="risk-badge {badge_class}">'
            f'{level} — {result["allowed_count"]}/14 tools allowed'
            f'</span>',
            unsafe_allow_html=True
        )

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score",      f"{result['risk_score']:.2f}")
        c2.metric("Modality",        result['modality'].upper())
        c3.metric("Tools Allowed",   f"{result['allowed_count']}/14")
        c4.metric("Tools Blocked",   f"{result['blocked_count']}/14")

        st.markdown("")

        # Two columns — signals + tools
        left, right = st.columns([1, 1])

        with left:
            st.markdown("#### Detection signals")
            sigs = result['signals']

            if result['modality'] == 'image':
                vis  = sigs.get('min_vis', 'N/A')
                mis  = sigs.get('mismatch', 'N/A')
                vf   = sigs.get('vis_flag', False)
                mf   = sigs.get('mis_flag', False)
                kw   = sigs.get('keyword', None)

                st.markdown(f"""
                <div class="signal-box">
                  <b>Visibility score</b>: {vis}
                  <span class="{'flag-fired' if vf else 'flag-clear'}">
                    {'🚨 Low visibility detected' if vf else '✅ Normal visibility'}
                  </span>
                </div>
                <div class="signal-box">
                  <b>Cross-modal mismatch</b>: {mis}
                  <span class="{'flag-fired' if mf else 'flag-clear'}">
                    {'🚨 Vision/OCR disagreement' if mf else '✅ Vision/OCR agree'}
                  </span>
                </div>
                <div class="signal-box">
                  <b>Injection keyword</b>:
                  <span class="{'flag-fired' if kw else 'flag-clear'}">
                    {'🚨 ' + str(kw) if kw else '✅ None found'}
                  </span>
                </div>
                <div class="signal-box">
                  <b>OCR text</b>: {sigs.get('ocr_text','')[:80]}...
                </div>
                <div class="signal-box">
                  <b>Vision description</b>: {sigs.get('vision_desc','')[:80]}
                </div>
                """, unsafe_allow_html=True)

            elif result['modality'] == 'audio':
                kw = sigs.get('keyword', None)
                st.markdown(f"""
                <div class="signal-box">
                  <b>Transcript</b>: {sigs.get('transcript','')[:120]}...
                </div>
                <div class="signal-box">
                  <b>Keyword score</b>: {sigs.get('kw_score', 0)}
                  <span class="{'flag-fired' if sigs.get('kw_score',0)>0 else 'flag-clear'}">
                    {'🚨 ' + str(kw) if kw else '✅ None found'}
                  </span>
                </div>
                <div class="signal-box">
                  <b>Semantic suspicion</b>: {sigs.get('sem_score', 0)}
                </div>
                """, unsafe_allow_html=True)

            else:
                kw = sigs.get('keyword', None)
                st.markdown(f"""
                <div class="signal-box">
                  <b>Input text</b>: {sigs.get('text','')[:120]}...
                </div>
                <div class="signal-box">
                  <b>Keyword score</b>: {sigs.get('kw_score', 0)}
                  <span class="{'flag-fired' if sigs.get('kw_score',0)>0 else 'flag-clear'}">
                    {'🚨 ' + str(kw) if kw else '✅ None found'}
                  </span>
                </div>
                <div class="signal-box">
                  <b>Semantic suspicion</b>: {sigs.get('sem_score', 0)}
                </div>
                """, unsafe_allow_html=True)

        with right:
            st.markdown("#### Tool restriction")

            if result['allowed_tools']:
                allowed_html = "".join([
                    f'<span class="tool-chip tool-allowed">{t}</span>'
                    for t in result['allowed_tools']
                ])
                st.markdown(
                    f"**✅ Allowed ({result['allowed_count']})**<br>{allowed_html}",
                    unsafe_allow_html=True
                )

            st.markdown("")

            if result['blocked_tools']:
                blocked_html = "".join([
                    f'<span class="tool-chip tool-blocked">{t}</span>'
                    for t in result['blocked_tools']
                ])
                st.markdown(
                    f"**🚫 Blocked ({result['blocked_count']})**<br>{blocked_html}",
                    unsafe_allow_html=True
                )

        # Risk bar
        st.markdown("")
        st.markdown("#### Risk score")
        risk_pct = int(result['risk_score'] * 100)
        color = (
            "#3B6D11" if risk_pct <= 25 else
            "#185FA5" if risk_pct <= 50 else
            "#854F0B" if risk_pct <= 75 else
            "#993C1D"
        )
        st.markdown(f"""
        <div style="background:#F0EEE8;border-radius:6px;height:12px;overflow:hidden">
          <div style="width:{risk_pct}%;height:100%;background:{color};border-radius:6px;
                      transition:width 0.5s ease"></div>
        </div>
        <p style="font-size:12px;color:#888;margin-top:4px">
          {risk_pct}% risk — {level}
        </p>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.caption(
    "M-SHIELD · A Risk-Aware Multimodal Defense Framework · "
    "Innovations: Visibility Trust Scoring · Cross-Modal Agreement · Risk-Aware Tool Restriction"
)
