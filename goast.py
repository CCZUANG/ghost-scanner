import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器 (2026)", page_icon="👻", layout="wide")

# 初始化 Session State (記憶還原系統)
if 'scan_limit' not in st.session_state: st.session_state.scan_limit = 600 
if 'min_vol_m' not in st.session_state: st.session_state.min_vol_m = 10
if 'dist_threshold' not in st.session_state: st.session_state.dist_threshold = 8.0
if 'u_sensitivity' not in st.session_state: st.session_state.u_sensitivity = 30

if 'backup' not in st.session_state:
    st.session_state.backup = {
        'scan_limit': 600, 'min_vol_m': 10, 'dist_threshold': 8.0, 'u_sensitivity': 30
    }

def handle_u_logic_toggle():
    """連動邏輯：啟動時備份設定，關閉時秒速還原"""
    if st.session_state.u_logic_key:
        st.session_state.backup.update({
            'scan_limit': st.session_state.scan_limit,
            'min_vol_m': st.session_state.min_vol_m,
            'dist_threshold': st.session_state.dist_threshold,
            'u_sensitivity': st.session_state.u_sensitivity
        })
        st.session_state.scan_limit = 600
        st.session_state.min_vol_m = 1
        st.session_state.dist_threshold = 50.0
        st.session_state.u_sensitivity = 60
    else:
        st.session_state.scan_limit = st.session_state.backup['scan_limit']
        st.session_state.min_vol_m = st.session_state.backup['min_vol_m']
        st.session_state.dist_threshold = st.session_state.backup['dist_threshold']
        st.session_state.u_sensitivity = st.session_state.backup['u_sensitivity']

st.title("👻 幽靈策略掃描器")
st.caption(f"📅 台灣時間：{datetime.now().strftime('%Y-%m-%d %H:%M')} (2026年)")

# --- 2. 核心策略導引區 (Step 1-3 文字完全回歸) ---
with st.expander("📖 幽靈策略：動態蝴蝶演化步驟 (詳細準則)", expanded=True):
    col_step1, col_step2, col_step3 = st.columns(3)
    
    with col_step1:
        st.markdown("### 第一步：建立試探部位 (Rule 1)")
        st.markdown("""
        **🚀 啟動時機**：放量突破關鍵壓力或回測支撐成功時。  
        **動作**：買進 低價位 Call + 賣出 高一階 Call (**多頭價差**)。  
        **成功指標**：股價站穩成本區，$\Delta$ (Delta) 隨價格上升而穩定增加。  
        **❌ 失敗判定**：2 交易日橫盤或跌破支撐 / 總損失超過 3 點。
        """)
        
    with col_step2:
        st.markdown("### 第二步：動能加碼 (Rule 2)")
        st.markdown("""
        **🚀 啟動時機**：當價差已產生「浮盈」，且股價衝向賣出價位時。  
        **動作**：加買 **更高一階的 Call**。  
        **成功指標**：IV 顯著擴張（**水結成冰**），部位因波動迅速膨脹。  
        **❌ 失敗判定**：動能衰竭或 IV 下降（冰塊融化）。
        """)
        
    with col_step3:
        st.markdown("### 第三步：轉化蝴蝶 (退出方案)")
        st.markdown("""
        **🚀 啟動時機**：股價強勢漲破加碼價，且市場出現過熱訊號時。  
        **動作**：**再加賣一張中間價位的 Call** (總計賣出兩張)。  
        **成功指標**：型態轉為 **蝴蝶型態 (+1/-2/+1)**，達成負成本。  
        **❌ 失敗判定**：爆量不漲或價格遠超最高階。
        """)

    st.info("💡 **核心注意事項**：Step 2 重點在於 IV 擴張。只有在部位已「證明你是對的」時才能執行 Rule 2 加碼。")

st.markdown("---")

# --- 3. 側邊欄 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio("市場", ["S&P 500", "NASDAQ 100", "🔥 全火力"], index=2)
enable_u_logic = st.sidebar.checkbox("✅ 啟動 4小時 U型戰法連動", value=False, key='u_logic_key', on_change=handle_u_logic_toggle)
scan_limit = st.sidebar.slider("掃描數量", 50, 600, key='scan_limit')

st.sidebar.header("🛡️ 趨勢濾網")
check_daily_ma60_up = st.sidebar.checkbox("✅ 日線 60MA 向上", value=True)
check_price_above_daily_ma60 = st.sidebar.checkbox("✅ 股價 > 日線 60MA", value=True)

st.sidebar.header("⚙️ 基礎篩選")
hv_threshold = st.sidebar.slider("HV Rank 門檻", 10, 100, 30)
min_vol_m = st.sidebar.slider("最小日均量 (百萬股)", 1, 100, key='min_vol_m') 
dist_threshold = st.sidebar.slider("距離 MA60 範圍 (%)", 0.0, 50.0, key='dist_threshold', step=0.5)

if enable_u_logic:
    u_sensitivity = st.sidebar.slider("U型敏感度", 20, 60, key='u_sensitivity')
    min_curvature = st.sidebar.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")
else:
    u_sensitivity, min_curvature = 30, 0.003
max_workers = st.sidebar.slider("🚀 平行核心數", 1, 32, 16)

# --- 4. 產業翻譯字典 ---
INDUSTRY_MAP = {
    "technology": "科技", "software": "軟體服務", "semiconductors": "半導體",
    "financial": "金融銀行", "healthcare": "醫療保健", "biotechnology": "生物科技",
    "energy": "能源", "industrials": "工業製造", "consumer cyclical": "循環性消費",
    "consumer defensive": "防禦性消費", "utilities": "公用事業", "real estate": "房地產",
    "communication services": "通訊服務", "basic materials": "基礎原物料",
    "entertainment": "影視娛樂", "internet content": "網路內容", "auto": "汽車產業",
    "retail": "零售通路", "aerospace": "航太軍工", "banks": "銀行業"
}

def translate_industry(eng):
    if not eng or eng == "N/A": return "未知"
    target = eng.lower()
    for key, val in INDUSTRY_MAP.items():
        if key in target: return val
    return eng

# --- 5. 核心繪圖函數 ---
def plot_interactive_chart(symbol):
    stock = yf.Ticker(symbol)
    tab1, tab2, tab3 = st.tabs(["🗓️ 周線", "📅 日線", "⏱️ 4H"])
    layout = dict(xaxis_rangeslider_visible=False, height=600, margin=dict(l=10, r=10, t=50, b=50), legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"), dragmode='pan')
    config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}

    try:
        with tab1:
            df = stock.history(period="5y", interval="1wk")
            df['MA60'] = df['Close'].rolling(60).mean()
            fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='周K'),
                             go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3))])
            fig.update_layout(title=dict(text=f"{symbol} 周線", x=0.02), **layout)
            st.plotly_chart(fig, use_container_width=True, config=config)
        with tab2:
            df = stock.history(period="2y")
            df['MA60'] = df['Close'].rolling(60).mean
