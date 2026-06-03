import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import urllib.request

# ── 페이지 설정 (반드시 첫 번째 st 호출) ─────────────────────
st.set_page_config(
    page_title="폐암 환자 군집 분석 시스템",
    page_icon="🫁",
    layout="centered",
)

# ── 한글 폰트 설정 ────────────────────────────────────────────
@st.cache_resource
def setup_korean_font():
    """NanumGothic 폰트를 다운로드하고 matplotlib에 등록합니다."""
    font_path = "/tmp/NanumGothic.ttf"

    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/nanum-fonts/raw/main/fonts/NanumGothic/NanumGothic-Regular.ttf"
        try:
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            # 다운로드 실패 시 시스템 한글 폰트 탐색
            for f in fm.findSystemFonts():
                fname = os.path.basename(f).lower()
                if any(k in fname for k in ["nanum", "malgun", "gulim", "dotum", "batang", "gothic"]):
                    prop = fm.FontProperties(fname=f)
                    plt.rcParams["font.family"] = prop.get_name()
                    plt.rcParams["axes.unicode_minus"] = False
                    return prop
            return None

    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return prop

korean_font = setup_korean_font()

# ── CSS 스타일 ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-desc {
        color: #555;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.3rem;
    }
    .result-box-green {
        background: #e8f5e9;
        border-left: 4px solid #43a047;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        color: #2e7d32;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .result-box-red {
        background: #fce4ec;
        border-left: 4px solid #e53935;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        color: #b71c1c;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .result-box-yellow {
        background: #fff8e1;
        border-left: 4px solid #fdd835;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        color: #f57f17;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .legend-text {
        color: #444;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    div[data-testid="stNumberInput"] label {
        font-size: 0.85rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# ── 샘플 데이터 생성 ──────────────────────────────────────────
@st.cache_data
def generate_sample_data(n=80, seed=42):
    rng = np.random.default_rng(seed)

    c0 = pd.DataFrame({
        "나이":   rng.normal(35, 6, n // 3).clip(20, 55),
        "흡연량": rng.normal(3,  2, n // 3).clip(0, 10),
        "음주량": rng.normal(1,  1, n // 3).clip(0,  5),
    })
    c1 = pd.DataFrame({
        "나이":   rng.normal(55, 8, n // 3).clip(35, 75),
        "흡연량": rng.normal(20, 5, n // 3).clip(10, 35),
        "음주량": rng.normal(5,  2, n // 3).clip(2,  10),
    })
    c2 = pd.DataFrame({
        "나이":   rng.normal(45, 7, n - 2*(n//3)).clip(25, 65),
        "흡연량": rng.normal(10, 3, n - 2*(n//3)).clip(3, 20),
        "음주량": rng.normal(3,  1, n - 2*(n//3)).clip(1,  6),
    })

    return pd.concat([c0, c1, c2], ignore_index=True)


# ── KMeans 학습 ───────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["나이", "흡연량", "음주량"]])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    centers = km.cluster_centers_
    order = np.argsort(centers[:, 1])       # 흡연량 기준 오름차순
    mapping = {order[0]: 0, order[1]: 2, order[2]: 1}
    labels_mapped = np.array([mapping[l] for l in labels])

    return km, scaler, labels_mapped, mapping


# ── 예측 함수 ─────────────────────────────────────────────────
def predict_cluster(km, scaler, mapping, age, smoke, drink):
    x_scaled = scaler.transform([[age, smoke, drink]])
    raw = km.predict(x_scaled)[0]
    return mapping[raw]


# ── 시각화 ────────────────────────────────────────────────────
def plot_clusters(df, labels, patient_smoke, patient_drink, patient_cluster):
    COLORS = {0: "#26a69a", 1: "#fdd835", 2: "#4a148c"}
    LABEL_KO = {0: "매우 건강군", 1: "위험군", 2: "건강군"}
    fp = korean_font  # FontProperties or None

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for cl in [0, 1, 2]:
        mask = labels == cl
        ax.scatter(
            df["흡연량"][mask], df["음주량"][mask],
            c=COLORS[cl], s=60, alpha=0.85, zorder=2,
            label=f"{cl}번 군집 ({LABEL_KO[cl]})"
        )

    ax.scatter(
        patient_smoke, patient_drink,
        marker="*", s=350, c="#1565c0", zorder=5,
        label=f"현재 환자 (→ {patient_cluster}번 군집)"
    )

    kw = {"fontproperties": fp} if fp else {"fontsize": 10}
    ax.set_xlabel("흡연량", **(kw if fp else {"fontsize": 10}))
    ax.set_ylabel("음주량", **(kw if fp else {"fontsize": 10}))
    ax.set_title("군집 시각화", fontsize=12, fontweight="bold",
                 **({"fontproperties": fp} if fp else {}))
    ax.legend(prop=fp if fp else None, fontsize=None if fp else 8,
              loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
#  메인 UI
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">🫁 폐암 환자 군집 분석 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-desc">AI가 환자의 특성을 분석하여<br>어떤 군집(유형)에 속하는지 예측합니다.</div>', unsafe_allow_html=True)
st.divider()

df = generate_sample_data()
km, scaler, labels, mapping = train_model(df)

st.markdown('<div class="section-title">📋 환자 정보 입력</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age   = st.number_input("나이",   min_value=0.0, max_value=120.0, value=50.0, step=1.0, format="%.2f")
with col2:
    smoke = st.number_input("흡연량", min_value=0.0, max_value=100.0, value=10.0, step=1.0, format="%.2f")
with col3:
    drink = st.number_input("음주량", min_value=0.0, max_value=100.0, value=5.0,  step=1.0, format="%.2f")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 군집 분석하기", use_container_width=True):
    cluster = predict_cluster(km, scaler, mapping, age, smoke, drink)

    BOX_CLASS    = {0: "result-box-green", 1: "result-box-red",   2: "result-box-green"}
    CLUSTER_NAME = {0: "0번 군집",          1: "1번 군집",          2: "2번 군집"}

    st.markdown(
        f'<div class="{BOX_CLASS[cluster]}">이 환자는 {CLUSTER_NAME[cluster]}에 속합니다.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="legend-text">0번은 매우 건강군, 1번은 위험군, 2번은 건강군입니다.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    fig = plot_clusters(df, labels, smoke, drink, cluster)
    st.pyplot(fig)
    plt.close(fig)
