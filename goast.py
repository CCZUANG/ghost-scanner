import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器 (題材快搜版)", page_icon="👻", layout="wide")

st.title("👻 幽靈策略掃描器 (題材快搜版)")
st.write("""
**策略目標**：以 **HV 低波動** 排序，鎖定 **日線多頭 + 4H U型** 標的，並提供 **一鍵查詢題材與風險** 功能。
""")

# --- 2. 側邊欄：參數設定區 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio(
    "選擇掃描市場", 
    ["S&P 500 (大型股)", "NASDAQ 100 (科技股)", "🔥 全火力 (兩者全掃)"],
    index=2
)
scan_limit = st.sidebar.slider("掃描數量 (前 N 大)", 50, 600, 200)

# --- 日線趨勢濾網 ---
st.sidebar.header("🛡️ 日線趨勢濾網")
check_daily_ma60_up = st.sidebar.checkbox("✅ 必須：日線 60MA 向上", value=True)
check_price_above_daily_ma60 = st.sidebar.checkbox("✅ 必須：股價 > 日線 60MA", value=True)

st.sidebar.header("⚙️ 基礎篩選")
hv_threshold = st.sidebar.slider("HV Rank 門檻 (越低越好)", 10, 100, 65)
min_vol_m = st.sidebar.slider("最小日均量 (百萬股)", 1, 20, 3) 
min_volume_threshold = min_vol_m * 1000000

st.sidebar.header("📈 4小時 U型戰法")
enable_u_logic = st.sidebar.checkbox("✅ 啟用「U型數學擬合」", value=True)
dist_threshold = st.sidebar.slider("距離 4H 60MA 範圍 (%)", 0.0, 50.0, 8.0, step=0.5)

if enable_u_logic:
    u_sensitivity = st.sidebar.slider("U型敏感度 (Lookback)", 20, 60, 30)
    min_curvature = st.sidebar.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")
else:
    u_sensitivity = 30
    min_curvature = 0.003

st.sidebar.markdown("---")
max_workers = st.sidebar.slider("🚀 平行運算核心數", 1, 32, 16)

# --- 產業翻譯字典 (擴充版) ---
# 將 key 全部轉為小寫以利比對
INDUSTRY_MAP = {
    "technology": "科技業",
    "software": "軟體",
    "semiconductors": "半導體",
    "financial": "金融",
    "banks": "銀行",
    "credit": "信貸",
    "healthcare": "醫療保健",
    "biotechnology": "生物科技",
    "consumer cyclical": "非必需消費",
    "auto": "汽車",
    "energy": "能源",
    "oil": "石油",
    "industrials": "工業",
    "aerospace": "航太軍工",
    "communication": "通訊",
    "internet": "網路",
    "utilities": "公用事業",
    "real estate": "房地產",
    "reit": "房地產信託",
    "basic materials": "原物料",
    "entertainment": "娛樂",
    "beverages": "飲料",
    "retail": "零售",
    "insurance": "保險",
    "telecom": "電信",
    "asset management": "資產管理"
}

def translate_industry(eng_industry):
    if not eng_industry or eng_industry == "N/A":
        return "未知"
    
    # 轉小寫並去除前後空白
    target = str(eng_industry).lower().strip()
    
    # 1. 嘗試完全匹配
    if target in INDUSTRY_MAP:
        return INDUSTRY_MAP[target]
    
    # 2. 嘗試部分關鍵字匹配 (只要包含關鍵字就翻譯)
    for key, value in INDUSTRY_MAP.items():
        if key in target:
            return value
            
    # 3. 真的翻不出來，回傳原文的首字大寫
    return target.title()

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
        df_1h = stock.history(period="6mo", interval="1h")
        if len(df_1h) < 240: return None

        # --- A. 日線級別處理 ---
        df_daily_synth = df_1h.resample('D').agg({
            'Volume': 'sum',
            'Close': 'last'
        }).dropna()
        
        df_daily_synth['MA60'] = df_daily_synth['Close'].rolling(window=60).mean()
        
        if len(df_daily_synth) < 60: return None
        
        daily_ma60_now = df_daily_synth['MA60'].iloc[-1]
        daily_ma60_prev = df_daily_synth['MA60'].iloc[-2]
        current_price_daily = df_daily_synth['Close'].iloc[-1]

        if check_daily_ma60_up and daily_ma60_now <= daily_ma60_prev: return None
        if check_price_above_daily_ma60 and current_price_daily < daily_ma60_now: return None

        avg_volume = df_daily_synth['Volume'].rolling(window=20).mean().iloc[-1]
        if avg_volume < vol_threshold: return None

        close_daily = df_daily_synth['Close']
        log_ret = np.log(close_daily / close_daily.shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        
        current_hv = vol_30d.iloc[-1]
        min_hv = vol_30d.min()
        max_hv = vol_30d.max()
        if max_hv == min_hv: return None
        hv_rank = ((current_hv - min_hv) / (max_hv - min_hv)) * 100
        
        if hv_rank > hv_threshold: return None

        # --- B. 4小時級別處理 ---
        df_4h = df_1h.resample('4h').agg({
            'Close': 'last', 
            'Volume': 'sum'
        }).dropna()
        
        if len(df_4h) < 60: return None

        df_4h['MA60'] = df_4h['Close'].rolling(window=60).mean()
        ma_segment = df_4h['MA60'].iloc[-u_sensitivity:]
        if ma_segment.isnull().values.any() or len(ma_segment) < u_sensitivity: return None
        
        current_price_4h = df_4h['Close'].iloc[-1]
        ma60_now_4h = ma_segment.iloc[-1]
        dist_pct = ((current_price_4h - ma60_now_4h) / ma60_now_4h) * 100

        if abs(dist_pct) > dist_threshold: return None 
        
        # --- C. U 型檢測 ---
        u_score = 0
        curvature = 0

        if enable_u_logic:
            is_u_shape, curv = analyze_u_shape(ma_segment)
            if not is_u_shape: return None
            if curv < min_curvature: return None
            curvature =
