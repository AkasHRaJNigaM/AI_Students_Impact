import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

/* App background: subtle radial glow */
.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(124,111,242,0.16) 0%, transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(46,196,182,0.10) 0%, transparent 40%),
        #0E1117;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #11141D;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stRadio > label { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 4px;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(124,111,242,0.12);
}

/* Hero */
.hero-badge {
    display:inline-block; padding: 4px 12px; border-radius: 999px;
    background: linear-gradient(90deg, rgba(124,111,242,0.18), rgba(46,196,182,0.18));
    border: 1px solid rgba(124,111,242,0.35);
    font-size: 0.78rem; color:#C9C4FF; margin-bottom: 10px; letter-spacing: 0.02em;
}
.hero-title {
    font-size: 2.4rem; font-weight: 700; line-height: 1.15;
    background: linear-gradient(90deg, #F5F3FF 20%, #B9B2FF 60%, #7EE8DA 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    margin-bottom: 6px;
}
.hero-sub { color: #9AA0B4; font-size: 1.02rem; max-width: 680px; }

/* Cards */
.card {
    background: linear-gradient(160deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 18px 20px;
    height: 100%;
}
.card h4 { margin: 0 0 6px 0; font-size: 0.95rem; color: #C9CDDA;}
.card .big { font-size: 1.7rem; font-weight: 700; color: #F5F3FF; font-family: 'Space Grotesk', sans-serif;}
.card .delta-pos { color: #6EE7B7; font-size: 0.85rem; }
.card .delta-neg { color: #FCA5A5; font-size: 0.85rem; }
.card .caption { color: #7A8096; font-size: 0.8rem; margin-top: 4px; }

.section-tag {
    display:inline-block; font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase;
    color:#7EE8DA; background: rgba(46,196,182,0.10); border:1px solid rgba(46,196,182,0.28);
    padding:3px 10px; border-radius:999px; margin-bottom:8px;
}

.pill {
    display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600;
}
.pill-low { background: rgba(110,231,183,0.15); color:#6EE7B7; border:1px solid rgba(110,231,183,0.4);}
.pill-medium { background: rgba(250,204,21,0.15); color:#FACC15; border:1px solid rgba(250,204,21,0.4);}
.pill-high { background: rgba(252,165,165,0.15); color:#FCA5A5; border:1px solid rgba(252,165,165,0.4);}

hr.soft { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.4rem 0; }

.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(90deg, #7C6FF2, #5B8DEF);
    color: white; border: none; border-radius: 10px; font-weight: 600;
    padding: 0.55rem 1.4rem;
}
.stButton > button:hover, .stFormSubmitButton > button:hover { opacity: 0.92; }

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(badge: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-badge">{badge}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-sub">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def section_tag(text: str):
    st.markdown(f'<span class="section-tag">{text}</span>', unsafe_allow_html=True)


def metric_card(label: str, value: str, caption: str = "", delta: str = "", positive: bool = True):
    delta_html = ""
    if delta:
        cls = "delta-pos" if positive else "delta-neg"
        delta_html = f'<div class="{cls}">{delta}</div>'
    st.markdown(
        f"""
        <div class="card">
            <h4>{label}</h4>
            <div class="big">{value}</div>
            {delta_html}
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_pill(level: str) -> str:
    cls = {"Low": "pill-low", "Medium": "pill-medium", "High": "pill-high"}.get(level, "pill-medium")
    return f'<span class="pill {cls}">{level}</span>'


PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_SEQUENCE = ["#7C6FF2", "#5B8DEF", "#2EC4B6", "#FACC15", "#FCA5A5"]
