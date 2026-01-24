import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器 (Pro+)", page_icon="👻", layout="wide")

st.title("👻 幽靈策略掃描器 (Pro+)")
st.write("""
**策略目標**：尋找「日線趨勢向上」且 **「4小時 60MA 剛形成微笑曲線 (翻揚)」** 的起漲點。
""")

# --- 2. 側邊欄：參數設定區 ---
st.sidebar.header("⚙️ 基礎篩選")
scan_limit = st.sidebar.slider("1. 掃描數量 (前 N 大)", 50, 500, 100)
hv_threshold = st.sidebar.slider("2. HV Rank 門檻 (低於多少)", 10, 80, 50, help="為了抓反轉型態，波動率可以稍微放寬")
min_vol_m = st.sidebar.slider("3. 最小日均量 (百萬股)", 1, 20, 3) 
min_volume_threshold = min_vol_m * 1000000

st.sidebar.header("📈 4小時 60MA 戰法")
only_ma_flip = st.sidebar.checkbox("✅ 嚴格篩選「微笑轉折」", value=True, help="只顯示 MA60 呈現 U 型反轉 (左跌右漲) 的股票")

# 這裡已保留您的需求：上限設為 50.0
dist_threshold = st.sidebar.slider("🎯 距離 60MA 範圍 (%)", 0.0, 50.0, 5.0, step=0.5, help="股價距離 60MA 多近？(上限已放寬至 50%)")

st.sidebar.markdown("---")
st.sidebar.info("💡 **圖形辨識邏輯**：\n程式會檢查過去 5 根 4H K棒的均線走勢，尋找「先跌、後平、再勾起」的 U 型結構。")

# --- 3. 核心函數 ---

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0"}
    try:
        url = '
