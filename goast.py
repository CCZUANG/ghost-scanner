import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器 (期權防呆版)", page_icon="👻", layout="wide")

st.title("👻 幽靈策略掃描器 (期權防呆版)")
st.write("""
**策略目標**：尋找 **S&P 500 / NASDAQ 100** 中，符合 **U型反轉** 且 **確認有期權** 的標的。
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
min_vol_m = st.sidebar.slider("最小日均量 (百萬股)", 1, 20, 3) 
min_volume_threshold = min_vol_m * 1000000

st.sidebar.header("📈 4小時 U型戰法")
dist_threshold = st.sidebar.slider("距離 60MA 範圍 (%)", 0.0, 50.0, 8.0, step=0.5)
u_sensitivity = st.sidebar.slider("U型敏感度 (Lookback)", 20, 60, 30)
min_curvature = st.sidebar.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")

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
        for df in dfs:
            if 'Ticker' in df.columns:
                tickers = df['Ticker'].tolist()
                return [t.replace('.', '-') for t in tickers]
            elif 'Symbol' in df.columns:
                tickers = df['Symbol'].tolist()
                return [t.replace('.', '-') for t in tickers]
        return []
    except:
        return []

def get_combined_tickers(choice, limit):
    sp500 = []
    nasdaq = []
    
    if "S&P" in choice or "全火力" in choice:
        sp500 = get_sp500_tickers()
    
    if "NASDAQ" in choice or "全火力" in choice:
        nasdaq = get_nasdaq100_tickers()
    
    combined = list(set(sp500 + nasdaq))
    
    if not combined:
        return ['TSM', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'PLTR', 'LUNR', 'COIN', 'MSTR', 'QQQ', 'SPY']
    
    return combined[:limit]

def analyze_u_shape(ma_series):
    try:
        y = ma_series.values
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs
        
        if a <= 0: return False, 0
        
        vertex_x = -b / (2 * a)
        len_window = len(y)
        
        if not (len_window * 0.3 <= vertex_x <= len_window * 1.1):
            return False, a
            
        current_slope = y[-1] - y[-2]
        if current_slope <= 0: return False, a

        return True, a
    except:
        return False, 0

def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol)
        df_1h = stock.history(period="3mo", interval="1h")
        
        if len(df_1h) < 240: return None

        # 1. 計算日均量
        df_daily_synth = df_1h.resample('D').agg({'Volume': 'sum'})
        avg_volume = df_daily_synth['Volume'].rolling(window=20).mean().iloc[-1]
        
        if avg_volume < vol_threshold: return None

        # 2. 合成 4H K線
        df_4h = df_1h.resample('4h').agg({
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        if len(df_4h) < 60: return None

        df_4h['MA60'] = df_4h['Close'].rolling(window=60).mean()
        
        ma_segment = df_4h['MA60'].iloc[-u_sensitivity:]
        if ma_segment.isnull().values.any() or len(ma_segment) < u_sensitivity: return None
        
        # --- U 型檢測 ---
        is_u_shape, curvature = analyze_u_shape(ma_segment)
        
        if not is_u_shape: return None
        if curvature < min_curvature: return None
        
        current_price = df_4h['Close'].iloc[-1]
        ma60_now = ma_segment.iloc[-1]
        dist_pct = ((current_price - ma60_now) / ma60_now) * 100
        
        if abs(dist_pct) > dist_threshold: return None 

        # --- 【新增】最終防線：期權存在性檢查 ---
        # 只有當股票通過上述所有困難篩選後，才檢查這一步（為了節省時間）
        try:
            # 嘗試獲取期權到期日列表，如果為空或報錯，代表無期權
            if not stock.options: 
                return None
        except:
            return None

        # 計算排序分數
        u_score = (curvature * 1000) - (abs(dist_pct) * 0.5)

        return {
            "代號": symbol,
            "現價": round(current_price, 2),
            "4H 60MA": round(ma60_now, 2),
            "U型強度": round(curvature * 1000, 2),
            "乖離率": f"{round(dist_pct, 2)}%",
            "狀態": "✅ 完美微笑
