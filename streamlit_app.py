"""
Streamlit dashboard for the Smart Content Summarization System.
Connects to the Flask REST API backend to summarize text, documents, URLs, and YouTube videos.
Features a high-fidelity glassmorphic UI with animated floating background elements.
"""

import time
import json
from collections import Counter
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from utils.avatar_template import build_avatar_html

# Configure the API endpoint
API_URL = "http://localhost:5000"

# 1. Page Configuration
st.set_page_config(
    page_title="Smart Summarizer Dashboard",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS & Floating Background Injection
# We inject a clean radial gradient background with subtle, slow-moving blurred blobs
# creating a modern, professional SaaS landing page feel.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography Reset */
    html, body, [class*="css"], .stApp, p, span, div, h1, h2, h3, h4, h5, h6, label, button {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
    }

    /* Fixed Floating Animated Background */
    .floating-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        overflow: hidden;
        background: radial-gradient(circle at 50% 50%, rgba(248, 250, 252, 1) 0%, rgba(241, 245, 249, 1) 100%);
    }

    /* Dark Mode Compatible Background */
    @media (prefers-color-scheme: dark) {
        .floating-bg {
            background: radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 1) 0%, rgba(2, 6, 23, 1) 100%);
        }
    }

    /* Soft Animated Blobs */
    .bubble {
        position: absolute;
        border-radius: 50%;
        filter: blur(90px);
        opacity: 0.12;
        animation: float-around infinite ease-in-out;
    }

    @media (prefers-color-scheme: dark) {
        .bubble {
            opacity: 0.15;
            filter: blur(120px);
        }
    }

    .bubble-1 {
        width: 450px;
        height: 450px;
        background: #6366F1; /* Indigo */
        top: -10%;
        left: -5%;
        animation-duration: 25s;
    }

    .bubble-2 {
        width: 500px;
        height: 500px;
        background: #A855F7; /* Purple */
        bottom: -15%;
        right: -10%;
        animation-duration: 32s;
        animation-delay: -5s;
    }

    .bubble-3 {
        width: 350px;
        height: 350px;
        background: #06B6D4; /* Cyan */
        top: 40%;
        left: 35%;
        animation-duration: 28s;
        animation-delay: -10s;
    }

    .bubble-4 {
        width: 400px;
        height: 400px;
        background: #EC4899; /* Pink */
        bottom: 10%;
        left: -10%;
        animation-duration: 36s;
        animation-delay: -18s;
    }

    /* Floating animation path */
    @keyframes float-around {
        0%, 100% {
            transform: translate(0, 0) scale(1);
        }
        33% {
            transform: translate(40px, -60px) scale(1.08);
        }
        66% {
            transform: translate(-30px, 30px) scale(0.95);
        }
    }

    /* Make Streamlit main container transparent */
    .stApp {
        background: transparent !important;
    }

    /* Sidebar Glassmorphism */
    div[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.45) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(226, 232, 240, 0.8) !important;
    }

    @media (prefers-color-scheme: dark) {
        div[data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.55) !important;
            backdrop-filter: blur(20px) !important;
            border-right: 1px solid rgba(51, 65, 85, 0.5) !important;
        }
    }

    /* General Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 18px !important;
        padding: 1.75rem !important;
        box-shadow: 0 10px 30px 0 rgba(31, 38, 135, 0.03) !important;
        margin-bottom: 1.5rem !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 36px 0 rgba(31, 38, 135, 0.05) !important;
    }

    @media (prefers-color-scheme: dark) {
        .glass-card {
            background: rgba(30, 41, 59, 0.45) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.2) !important;
        }
        .glass-card:hover {
            box-shadow: 0 12px 36px 0 rgba(0, 0, 0, 0.25) !important;
        }
    }

    /* Style Text Areas and Inputs */
    div[data-testid="stTextArea"] textarea, 
    div[data-testid="stTextInput"] input,
    div[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.6) !important;
        border: 1px solid rgba(203, 213, 225, 0.8) !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    @media (prefers-color-scheme: dark) {
        div[data-testid="stTextArea"] textarea, 
        div[data-testid="stTextInput"] input,
        div[data-testid="stFileUploader"] {
            background-color: rgba(15, 23, 42, 0.5) !important;
            border: 1px solid rgba(71, 85, 105, 0.6) !important;
            color: #E2E8F0 !important;
        }
    }

    div[data-testid="stTextArea"] textarea:focus, 
    div[data-testid="stTextInput"] input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        background-color: rgba(255, 255, 255, 0.95) !important;
    }

    @media (prefers-color-scheme: dark) {
        div[data-testid="stTextArea"] textarea:focus, 
        div[data-testid="stTextInput"] input:focus {
            background-color: rgba(15, 23, 42, 0.85) !important;
        }
    }

    /* Buttons Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25) !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(99, 102, 241, 0.4) !important;
        color: white !important;
    }

    div.stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Premium Alert Box */
    div.stAlert {
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        background-color: rgba(245, 243, 255, 0.5) !important;
    }

    @media (prefers-color-scheme: dark) {
        div.stAlert {
            background-color: rgba(99, 102, 241, 0.08) !important;
            border: 1px solid rgba(99, 102, 241, 0.25) !important;
        }
    }

    /* Keyword Pills */
    .keyword-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.08);
        color: #4F46E5;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 7px 15px;
        border-radius: 9999px;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: default;
    }

    .keyword-pill:hover {
        background: rgba(99, 102, 241, 0.15);
        transform: translateY(-1.5px);
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.1);
    }

    @media (prefers-color-scheme: dark) {
        .keyword-pill {
            background: rgba(129, 140, 248, 0.12);
            color: #C7D2FE;
            border: 1px solid rgba(129, 140, 248, 0.25);
        }
        .keyword-pill:hover {
            background: rgba(129, 140, 248, 0.2);
            box-shadow: 0 4px 10px rgba(129, 140, 248, 0.15);
        }
    }

    /* Custom Title Styling */
    .dashboard-header {
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Clean metric layout */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #4F46E5 !important;
    }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetricValue"] {
            color: #C7D2FE !important;
        }
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600 !important;
        color: #64748B !important;
    }

    /* Hide Streamlit footer and menu clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
    </style>

    <div class="floating-bg">
      <div class="bubble bubble-1"></div>
      <div class="bubble bubble-2"></div>
      <div class="bubble bubble-3"></div>
      <div class="bubble bubble-4"></div>
    </div>
    """,
    unsafe_allow_html=True
)

# Cache requests session for efficiency
@st.cache_resource
def get_http_session() -> requests.Session:
    """
    Creates and caches a requests Session to handle API requests.
    """
    session = requests.Session()
    return session


# 3. Session State Initialization
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "url_input" not in st.session_state:
    st.session_state.url_input = ""
if "yt_input" not in st.session_state:
    st.session_state.yt_input = ""
if "mode" not in st.session_state:
    st.session_state.mode = "Extractive"
if "ratio" not in st.session_state:
    st.session_state.ratio = 0.2
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None


# Helper function to clear results
def clear_results():
    st.session_state.summary_result = None
    st.session_state.last_error = None


# 4. Main Interface Header
st.markdown(
    """
    <div style="text-align: center; margin-top: 1.5rem; margin-bottom: 2.5rem;">
        <h1 class="dashboard-header">Smart Content Summarizer</h1>
        <p style="color: #64748B; font-size: 1.15rem; font-weight: 500; margin-top: -0.25rem;">
            Summarize documents, web articles, raw text, and YouTube video transcripts in one click.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# 5. Sidebar Panel (Inputs and Parameters)
st.sidebar.markdown(
    """
    <div style="margin-top: 1rem; margin-bottom: 1.5rem;">
        <h2 style="font-weight: 800; font-size: 1.4rem; color: #1E293B; margin-bottom: 0.25rem; letter-spacing: -0.3px;">Configuration</h2>
        <div style="width: 40px; height: 4px; background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); border-radius: 2px;"></div>
    </div>
    """,
    unsafe_allow_html=True
)

# Input Mode Selector
input_mode = st.sidebar.radio(
    "Select Input Source",
    ["Paste Text", "Upload PDF/TXT", "YouTube URL", "Web URL"],
    on_change=clear_results
)

# Contextual Input Renderers (persisted in st.session_state)
payload_text = ""
uploaded_file = None

if input_mode == "Paste Text":
    payload_text = st.sidebar.text_area(
        "Enter Raw Text",
        placeholder="Type or paste your content here...",
        height=280,
        key="text_input"
    )

elif input_mode == "Upload PDF/TXT":
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF or TXT File (Max 5MB)",
        type=["pdf", "txt"],
        key="file_uploader_widget"
    )

elif input_mode == "YouTube URL":
    payload_text = st.sidebar.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="yt_input"
    )
    st.sidebar.caption(
        "⚠️ Requires the video to have **subtitles/captions** enabled. "
        "If the video has no captions, paste the content via **Paste Text** instead."
    )

elif input_mode == "Web URL":
    payload_text = st.sidebar.text_input(
        "Webpage URL",
        placeholder="https://example.com/article...",
        key="url_input"
    )

st.sidebar.markdown("<hr style='border: none; border-top: 1px solid rgba(226, 232, 240, 0.8); margin: 1.5rem 0;'/>", unsafe_allow_html=True)

# Summarization Mode
mode_choice = st.sidebar.radio(
    "Summarization Pipeline",
    ["Extractive", "Abstractive"],
    index=0 if st.session_state.mode == "Extractive" else 1,
    key="mode_radio"
)
st.session_state.mode = mode_choice

# Summary Ratio (Slider)
ratio_choice = st.sidebar.slider(
    "Summary Ratio (Keep fraction)",
    min_value=0.1,
    max_value=0.5,
    value=st.session_state.ratio,
    step=0.05,
    help="Higher ratios yield longer summaries (only applies to Extractive).",
    key="ratio_slider"
)
st.session_state.ratio = ratio_choice

# AI Voice Assistant configuration toggle
st.sidebar.markdown("<hr style='border: none; border-top: 1px solid rgba(226, 232, 240, 0.8); margin: 1.25rem 0;'/>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style="margin-top: -0.5rem; margin-bottom: 0.5rem;">
        <h3 style="font-weight: 800; font-size: 1.2rem; color: #1E293B; margin-bottom: 0.25rem; letter-spacing: -0.3px;">AI Voice Assistant</h3>
    </div>
    """,
    unsafe_allow_html=True
)
avatar_enabled = st.sidebar.checkbox(
    "Enable AI Avatar Reader 🤖",
    value=False,
    help="Enable an animated talking avatar to read out the summary with synchronized mouth movement."
)

st.sidebar.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

# Action Trigger Button
summarize_trigger = st.sidebar.button(
    "Generate Summary",
    use_container_width=True,
    type="primary"
)

# 6. API Request Orchestrator
if summarize_trigger:
    # Validation checks
    valid = True
    error_msg = ""
    
    if input_mode == "Upload PDF/TXT" and uploaded_file is None:
        valid = False
        error_msg = "Please upload a PDF or TXT file first."
    elif input_mode != "Upload PDF/TXT" and not payload_text.strip():
        valid = False
        error_msg = f"Please enter content or a valid URL for the {input_mode} mode."
        
    if not valid:
        st.session_state.last_error = error_msg
        st.session_state.summary_result = None
    else:
        st.session_state.last_error = None
        session = get_http_session()
        
        with st.spinner("Processing document and generating summary..."):
            try:
                # Perform call depending on the input source
                if input_mode == "Upload PDF/TXT":
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {
                        "mode": mode_choice.lower(),
                        "ratio": ratio_choice
                    }
                    response = session.post(
                        f"{API_URL}/upload",
                        files=files,
                        data=data,
                        timeout=90
                    )
                else:
                    payload = {
                        "text": payload_text.strip(),
                        "mode": mode_choice.lower(),
                        "ratio": ratio_choice
                    }
                    response = session.post(
                        f"{API_URL}/summarize",
                        json=payload,
                        timeout=90
                    )
                
                # Handle response
                if response.status_code == 200:
                    st.session_state.summary_result = response.json()
                else:
                    try:
                        err_json = response.json()
                        st.session_state.last_error = err_json.get("error", "An unknown error occurred on the server.")
                    except ValueError:
                        st.session_state.last_error = f"Server returned error code {response.status_code}: {response.text}"
                    st.session_state.summary_result = None

            except requests.exceptions.Timeout:
                st.session_state.last_error = (
                    "The request timed out (limit: 90s). This usually happens when "
                    "processing very large documents or when downloading a large model. "
                    "Please try a shorter text or check if the backend is responsive."
                )
                st.session_state.summary_result = None
            except requests.exceptions.ConnectionError:
                st.session_state.last_error = (
                    "Unable to connect to the Flask API backend at http://localhost:5000. "
                    "Make sure you have started the Flask server by running 'python app.py' in the background."
                )
                st.session_state.summary_result = None
            except Exception as e:
                st.session_state.last_error = f"Request failed: {str(e)}"
                st.session_state.summary_result = None


# 7. Main Dashboard Area Rendering
# Smart Error Display
if st.session_state.last_error:
    raw_err = st.session_state.last_error

    # Detect YouTube-specific error categories for friendly display
    yt_no_subs = any(k in raw_err.lower() for k in [
        "subtitles are disabled", "no transcript", "disabled subtitles",
        "no transcripts of any kind", "captions"
    ])
    yt_unavailable = any(k in raw_err.lower() for k in [
        "unavailable", "private", "deleted", "region-restricted"
    ])
    yt_rate_limit = "rate-limit" in raw_err.lower() or "rate_limit" in raw_err.lower()
    is_yt_error = (
        input_mode == "YouTube URL"
        or "youtube" in raw_err.lower()
        or "video id" in raw_err.lower()
        or "video (id:" in raw_err.lower()
        or "transcript" in raw_err.lower()
    )

    if is_yt_error and yt_no_subs:
        st.markdown(
            """
            <div style="background: rgba(254, 243, 199, 0.4); border: 1px solid #F59E0B;
                        backdrop-filter: blur(8px); border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <div style="font-size: 1.3rem; font-weight: 700; color: #B45309; margin-bottom: 0.5rem;">
                    📺 &nbsp;No Transcript Available
                </div>
                <p style="color: #78350F; margin: 0 0 1rem 0; font-size: 0.95rem; line-height: 1.5;">
                    This video does not have subtitles or captions enabled by the creator,
                    so there is no transcript to summarize.
                </p>
                <div style="color: #B45309; font-weight: 600; font-size: 0.9rem; border-top: 1px dashed rgba(245, 158, 11, 0.3); padding-top: 0.75rem;">
                    💡 <strong>Alternatives:</strong><br/>
                    &nbsp;&nbsp;• Switch to <em>Paste Text</em> mode and paste the transcript manually<br/>
                    &nbsp;&nbsp;• Try a different YouTube video that has captions enabled<br/>
                    &nbsp;&nbsp;• Check if the video has the <strong>CC</strong> icon in the YouTube player
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif is_yt_error and yt_unavailable:
        st.error("🔒 **Video Unavailable** — This video is private, deleted, or not accessible in your region. Please check the URL.")
    elif is_yt_error and yt_rate_limit:
        st.warning(
            "⏳ **YouTube Rate Limit** — YouTube temporarily blocked the transcript request. "
            "Wait a minute and try again, or paste the text manually via **Paste Text** mode."
        )
    elif is_yt_error:
        st.markdown(
            f"""
            <div style="background: rgba(254, 226, 226, 0.4); border: 1px solid #EF4444; backdrop-filter: blur(8px); border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <div style="font-weight: 700; color: #B91C1C; font-size: 1.15rem; margin-bottom: 0.4rem;">📺 YouTube Ingestion Failed</div>
                <p style="color: #7F1D1D; font-size: 0.95rem; margin: 0; line-height: 1.4;">{raw_err.split("Note:")[0].split("If this")[0].strip()}</p>
                <p style="color: #B91C1C; font-weight: 600; font-size: 0.9rem; margin: 0.75rem 0 0 0;">
                    👉 Try copy-pasting the text manually in the <strong>Paste Text</strong> tab.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error(raw_err)

# Results Dashboard
if st.session_state.summary_result:
    res = st.session_state.summary_result
    summary_text = res.get("summary", "")
    
    # Left / Right Split for Results & Statistics
    col_out_left, col_out_right = st.columns([7, 5], gap="large")
    
    with col_out_left:
        # Card 1: Summary Output
        st.markdown(
            """
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem;">
                    <h3 style="font-weight: 800; font-size: 1.4rem; color: #1E293B; margin: 0; letter-spacing: -0.3px;">Generated Summary</h3>
                    <span style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); color: white; font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 9999px; text-transform: uppercase;">Ready</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display the summary text beautifully inside st.info equivalent styled box
        st.info(summary_text)
        
        # Action button to download
        st.download_button(
            label="Download Summary (.txt)",
            data=summary_text,
            file_name="summary.txt",
            mime="text/plain",
            help="Download the summary as a text file.",
            use_container_width=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Card 2: Keywords
        keywords = res.get("keywords", [])
        if keywords:
            st.markdown(
                """
                <div class="glass-card">
                    <h3 style="font-weight: 800; font-size: 1.3rem; color: #1E293B; margin-bottom: 1.25rem; letter-spacing: -0.3px;">Key Topic Extractions</h3>
                """,
                unsafe_allow_html=True
            )
            
            pills = []
            for kw in keywords:
                pills.append(f'<span class="keyword-pill"># {kw}</span>')
            
            st.markdown("".join(pills), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_out_right:
        if avatar_enabled:
            # Render avatar HTML component
            avatar_html = build_avatar_html(summary_text)
            components.html(avatar_html, height=560, scrolling=False)

        # Card 3: Metrics
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="font-weight: 800; font-size: 1.3rem; color: #1E293B; margin-bottom: 1.25rem; letter-spacing: -0.3px;">Document Performance</h3>
            """,
            unsafe_allow_html=True
        )
        
        original_words = res.get("word_count", 0)
        summary_words = len(summary_text.split())
        compression_ratio = res.get("compression", "0% reduction")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label="Original Words",
                value=f"{original_words:,}"
            )
        with col_m2:
            st.metric(
                label="Summary Words",
                value=f"{summary_words:,}"
            )
            
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        
        st.metric(
            label="Text Reduction Rate",
            value=compression_ratio,
            help="Ratio of words removed by the summarization pipeline."
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Card 4: Entities Detected
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="font-weight: 800; font-size: 1.3rem; color: #1E293B; margin-bottom: 1.25rem; letter-spacing: -0.3px;">Entity Recognition Matrix</h3>
            """,
            unsafe_allow_html=True
        )
        
        entities_data = res.get("entities", {})
        entity_rows = []
        for ent_type, ent_list in entities_data.items():
            counts = Counter(ent_list)
            for entity, count in counts.items():
                entity_rows.append({
                    "Entity": entity,
                    "Type": ent_type,
                    "Frequency": count
                })
        
        if entity_rows:
            df_entities = pd.DataFrame(entity_rows)
            df_entities = df_entities.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
            st.dataframe(
                df_entities,
                use_container_width=True,
                hide_index=True,
                height=240
            )
        else:
            st.info("No notable named entities classified in the document.")
            
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Welcome / Empty State Board
    if not st.session_state.last_error:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 4.5rem 3rem !important; margin-top: 1.5rem;">
                <div style="font-size: 3.5rem; margin-bottom: 1.25rem;">✨</div>
                <h3 style="color: #1E293B; font-weight: 800; font-size: 1.6rem; letter-spacing: -0.4px; margin-bottom: 0.75rem;">
                    Configure and Begin
                </h3>
                <p style="color: #64748B; font-size: 1.05rem; max-width: 540px; margin: 0.5rem auto 2.25rem auto; line-height: 1.6; font-weight: 400;">
                    Select an input mode in the control panel sidebar, adjust the summarizer mode and ratios, and trigger processing to view the key takeaways.
                </p>
                <div style="display: flex; justify-content: center; gap: 2.5rem; flex-wrap: wrap;">
                    <div style="text-align: center; max-width: 130px;">
                        <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">📋</div>
                        <p style="font-size: 0.9rem; color: #475569; font-weight: 600; margin: 0;">Pasted Text</p>
                    </div>
                    <div style="text-align: center; max-width: 130px;">
                        <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">📂</div>
                        <p style="font-size: 0.9rem; color: #475569; font-weight: 600; margin: 0;">Documents</p>
                    </div>
                    <div style="text-align: center; max-width: 130px;">
                        <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">📺</div>
                        <p style="font-size: 0.9rem; color: #475569; font-weight: 600; margin: 0;">YouTube Videos</p>
                    </div>
                    <div style="text-align: center; max-width: 130px;">
                        <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🌐</div>
                        <p style="font-size: 0.9rem; color: #475569; font-weight: 600; margin: 0;">Web Articles</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
