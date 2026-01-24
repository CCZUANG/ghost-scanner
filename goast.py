import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器 (全功能版)", page_icon="👻", layout="wide")

st.title("👻 幽靈策略掃描器 (全功能版)")
st.write("""
**策略目標**：尋找 **低波動 (HV)** 且 **貼近 4H 60MA** 的標的，並支援 **U型反轉** 形態識別。
""")

# --- 2. 側邊欄：參數設定區 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio(
    "選擇掃描市場", 
    ["S&P 500 (大型股)", "NASDAQ 100 (科技股)", "🔥 全火力 (兩者全掃)"],
    index=2
)
scan_limit = st.sidebar.slider("掃描數量 (前 N 大)", 50, 600, 200)

st.sidebar.header("⚙️ 篩選條件")
# 【已修復】HV Rank 加回來了！
hv_threshold = st.sidebar.slider("HV Rank 門檻 (越低越好)", 10, 100, 60, help="數值越低代表波動越小 (結冰)，適合 Step 1 進場。")
min_vol_m = st.sidebar.slider("最小日均量 (百萬股)", 1, 20, 3) 
min_volume_threshold = min_vol_m * 1000000

# --- U型戰法開關 ---
st.sidebar.header("📈 4小時 U型戰法")
enable_u_logic = st.sidebar.checkbox("✅ 啟用「U型數學擬合」過濾", value=True, help="打勾：嚴格篩選完美 U 型。\n取消：只篩選乖離率，不看形狀。")

dist_threshold = st.sidebar.slider("距離 60MA 範圍 (%)", 0.0, 50.0, 8.0, step=0.5)

if enable_u_logic:
    u_sensitivity = st.sidebar.slider("U型敏感度 (Lookback)", 20, 60, 30)
    min_curvature = st.sidebar.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")
else:
    u_sensitivity = 30
    min_curvature = 0.003

st.sidebar.markdown("---")
max_workers = st.sidebar.slider("🚀 平行運算核心數", 1, 32, 16)

# --- 3. 核心函數 ---

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(response.text))[0]
        tickers = df['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except:
        return []

@st.cache_data(ttl=3600)
def get_nasdaq100_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=headers)
        dfs = pd.read_html(StringIO(response.text))
