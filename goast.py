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
hv_threshold = st.sidebar.slider("2. HV Rank 門檻", 10, 80, 50)
min_vol_m = st.sidebar.slider("3. 最小日均量 (百萬股)", 1, 20, 3) 
min_volume_threshold = min_vol_m * 1000000

st.sidebar.header("📈 4小時 60MA 戰法")
only_ma_flip = st.sidebar.checkbox("✅ 嚴格篩選「微笑轉折」", value=True)
dist_threshold = st.sidebar.slider("🎯 距離 60MA 範圍 (%)", 0.0, 50.0, 5.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.info("💡 **圖形辨識**：尋找 4H K線圖中，60MA 呈現 U 型反轉的標的。")

#
