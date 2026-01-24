import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器 (Pro)", page_icon="👻", layout="wide")

st.title("👻 幽靈策略掃描器 (Pro)")
st.write("""
**策略目標**：尋找「低波動 + 上漲趨勢 + 關鍵型態」的 S&P 500 標的。
""")

# --- 2. 側邊欄：參數設定區 ---
st.sidebar.header("⚙️ 參數設定")
scan_limit = st.sidebar.slider("1. 掃描數量 (前 N 大)", 50, 500, 100)
hv_threshold = st.sidebar.slider("2. HV Rank 門檻 (低於多少)", 10, 60, 40, help="放寬一點，讓我們用『型態』來過濾")
min_vol_m = st.sidebar.slider("3. 最小日均量 (百萬股)", 1, 20, 5)
min_volume_threshold = min_vol_m * 1000000

st.sidebar.markdown("---")
st.sidebar.markdown("""
**📊 型態說明：**
* **🧊 極度壓縮 (Squeeze)**：布林通道極窄，變盤在即 (Step 1 首選)。
* **📉 回測支撐 (Pullback)**：股價回測 20MA，低接機會。
* **📈 穩健上漲**：趨勢向上，無特殊型態。
""")

# --- 3. 核心函數 ---

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0"}
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, headers=headers)
        sp500_df = pd.read_html(StringIO(response.text))[0]
        tickers = sp500_df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except:
        # 備用清單，防止爬蟲失敗
        return ['TSM', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'PLTR', 'LUNR']

def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        
        if len(df) < 100: return None
        
        # A. 流動性過濾
        avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
        if avg_volume < vol_threshold: return None 
        
        # B. 技術指標計算
        close = df['Close']
        current_price = close.iloc[-1]
        
        # 1. 趨勢 (20MA)
        sma20 = close.rolling(window=20).mean().iloc[-1]
        trend_up = current_price > sma20
        
        # 2. 波動 (HV Rank)
        log_ret = np.log(close / close.shift(
