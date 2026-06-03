import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os, urllib.request, tempfile

# ── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(
    page_title="폐암 환자 군집 분석 시스템",
    page_icon="🫁",
    layout="centered",
)

# ── 한글 폰트 로드 ────────────────────────────────────────────
# Streamlit Cloud / Linux 환경 모두 대응
# 1순위: 번들된 NanumGothic (apt 설치 경로)
# 2순위: GitHub에서 직접 다운로드
# 3순위: 시스템 폰트 중 한글 지원 폰트
@st.cache_resource
def load_korean_font():
    candidates = [
        # Ubuntu/Streamlit Cloud apt 설치 경로
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        # macOS
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        # Windows
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # GitHub에서 NanumGothic 다운로드
    dl_path = os.path.join(tempfile.gettempdir(), "NanumGothic_dl.ttf")
    if not os.path.exists(dl_path):
        try:
            url = ("https://github.com/googlefonts/nanum-fonts/raw/main"
                   "/fonts/NanumGothic/NanumGothic-Regular.ttf")
            urllib.request.urlretrieve(url, dl_path)
        except Exception:
            return None
    return dl_path

FONT_PATH = load_korean_font()

def get_font_prop(size=10):
    if FONT_PATH:
        return fm.FontProperties(fname=FONT_PATH, size=size)
    return fm.FontProperties(size=size)

# rcParams도 설정 (축 tick 라벨 등에 적용)
if FONT_PATH:
    fm.fontManager.addfont(FONT_PATH)
    _name = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["font.family"] = _name
plt.rcParams["axes.unicode_minus"] = False

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.main-title  { font-size:2rem; font-weight:700; color:#1a1a2e; margin-bottom:.2rem; }
.sub-desc    { color:#555; font-size:.9rem; margin-bottom:1.5rem; }
.section-title { font-size:1.1rem; font-weight:600; color:#1a1a2e;
                 border-bottom:1px solid #e0e0e0; padding-bottom:.3rem; margin-bottom:.5rem; }
.box-green  { background:#e8f5e9; border-left:4px solid #43a047; border-radius:6px;
              padding:.7rem 1rem; color:#2e7d32; font-weight:500; margin-bottom:.5rem; }
.box-red    { background:#fce4ec; border-left:4px solid #e53935; border-radius:6px;
              padding:.7rem 1rem; color:#b71c1c; font-weight:500; margin-bottom:.5rem; }
.legend-txt { color:#444; font-size:.85rem; margin-top:.3rem; }
</style>
""", unsafe_allow_html=True)


# ── 데이터 ────────────────────────────────────────────────────
@st.cache_data
def generate_data(n=80, seed=42):
    rng = np.random.default_rng(seed)
    c0 = pd.DataFrame({"나이": rng.normal(35,6,n//3).clip(20,55),
                        "흡연량": rng.normal(3,2,n//3).clip(0,10),
                        "음주량": rng.normal(1,1,n//3).clip(0,5)})
    c1 = pd.DataFrame({"나이": rng.normal(55,8,n//3).clip(35,75),
                        "흡연량": rng.normal(20,5,n//3).clip(10,35),
                        "음주량": rng.normal(5,2,n//3).clip(2,10)})
    c2 = pd.DataFrame({"나이": rng.normal(45,7,n-2*(n//3)).clip(25,65),
                        "흡연량": rng.normal(10,3,n-2*(n//3)).clip(3,20),
                        "음주량": rng.normal(3,1,n-2*(n//3)).clip(1,6)})
    return pd.concat([c0,c1,c2], ignore_index=True)

@st.cache_resource
def train_model(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[["나이","흡연량","음주량"]])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    raw = km.fit_predict(X)
    order = np.argsort(km.cluster_centers_[:,1])
    mapping = {order[0]:0, order[1]:2, order[2]:1}
    return km, scaler, np.array([mapping[l] for l in raw]), mapping

def predict(km, scaler, mapping, age, smoke, drink):
    raw = km.predict(scaler.transform([[age, smoke, drink]]))[0]
    return mapping[raw]


# ── 차트 ─────────────────────────────────────────────────────
def make_chart(df, labels, p_smoke, p_drink, p_cluster):
    COLORS = {0:"#26a69a", 1:"#fdd835", 2:"#4a148c"}
    NAMES  = {0:"매우 건강군", 1:"위험군", 2:"건강군"}

    fig, ax = plt.subplots(figsize=(7,5), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fp10 = get_font_prop(10)
    fp8  = get_font_prop(8)
    fp12 = get_font_prop(12)

    for cl in [0,1,2]:
        m = labels == cl
        ax.scatter(df["흡연량"][m], df["음주량"][m],
                   c=COLORS[cl], s=60, alpha=.85, zorder=2,
                   label=f"{cl}번 군집 ({NAMES[cl]})")

    ax.scatter(p_smoke, p_drink, marker="*", s=350, c="#1565c0", zorder=5,
               label=f"현재 환자 (→ {p_cluster}번 군집)")

    ax.set_xlabel("흡연량", fontproperties=fp10)
    ax.set_ylabel("음주량", fontproperties=fp10)
    ax.set_title("군집 시각화", fontproperties=fp12, fontweight="bold")
    ax.grid(True, alpha=.3, linestyle="--")

    leg = ax.legend(prop=fp8, loc="upper left")

    # tick 라벨 폰트도 명시 적용
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(get_font_prop(9))

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🫁 폐암 환자 군집 분석 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-desc">AI가 환자의 특성을 분석하여<br>어떤 군집(유형)에 속하는지 예측합니다.</div>', unsafe_allow_html=True)
st.divider()

df = generate_data()
km, scaler, labels, mapping = train_model(df)

st.markdown('<div class="section-title">📋 환자 정보 입력</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: age   = st.number_input("나이",   0.0, 120.0, 50.0, 1.0, "%.2f")
with c2: smoke = st.number_input("흡연량", 0.0, 100.0, 10.0, 1.0, "%.2f")
with c3: drink = st.number_input("음주량", 0.0, 100.0,  5.0, 1.0, "%.2f")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 군집 분석하기", use_container_width=True):
    cl = predict(km, scaler, mapping, age, smoke, drink)
    BOX  = {0:"box-green", 1:"box-red",   2:"box-green"}
    NAME = {0:"0번 군집",  1:"1번 군집",  2:"2번 군집"}

    st.markdown(f'<div class="{BOX[cl]}">이 환자는 {NAME[cl]}에 속합니다.</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="legend-txt">0번은 매우 건강군, 1번은 위험군, 2번은 건강군입니다.</div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig = make_chart(df, labels, smoke, drink, cl)
    st.pyplot(fig)
    plt.close(fig)
