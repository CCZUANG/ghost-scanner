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
            curvature = curv
            u_score = (curvature * 1000) - (abs(dist_pct) * 0.5)
        else:
            u_score = -abs(dist_pct)

        # --- D. 期權檢查 ---
        try:
            if not stock.options: return None
        except:
            return None

        # --- E. 資訊豐富化 (產業中文 + 題材搜尋連結) ---
        industry_tw = "未知"
        earnings_date_str = "未知"

        try:
            # 1. 產業資訊
            info = stock.info
            raw_industry = info.get('industry', info.get('sector', 'N/A'))
            industry_tw = translate_industry(raw_industry)
            
            # 2. 財報日期
            cal = stock.calendar
            if cal and isinstance(cal, dict) and 'Earnings Date' in cal:
                earnings_date_str = cal['Earnings Date'][0].strftime('%m-%d')
            elif cal and isinstance(cal, dict) and 'Earnings High' in cal:
                 earnings_date_str = cal['Earnings High'][0].strftime('%m-%d')
            
        except:
            pass
        
        # 3. 生成 Google 搜尋連結 (題材與注意事項)
        # 關鍵字：股票代號 + 美股 + 題材 + 分析 + 風險
        search_query = f"{symbol}+美股+題材+風險+分析"
        search_url = f"https://www.google.com/search?q={search_query}"

        return {
            "代號": symbol,
            "HV Rank": round(hv_rank, 1),
            "現價": round(current_price_4h, 2),
            "4H 60MA": round(ma60_now_4h, 2),
            "乖離率": f"{round(dist_pct, 2)}%",
            "產業": industry_tw,
            "財報日": earnings_date_str,
            "題材搜尋": search_url, # 這是連結
            "_sort_score": u_score,
            "_dist_raw": abs(dist_pct)
        }
    except:
        return None

# --- 4. 主程式執行邏輯 ---

if st.button("🚀 啟動 Turbo 掃描", type="primary"):
    status_text = f"正在下載 {market_choice} 清單..."
    progress_bar = st.progress(0)
    
    with st.status(status_text, expanded=True) as status:
        target_tickers = get_combined_tickers(market_choice, scan_limit)
        
        status.write(f"🔥 Turbo 模式啟動！ (核心數: {max_workers})")
        status.write(f"🔍 目標: {len(target_tickers)} 檔 | 正在分析...")
        
        results = []
        completed_count = 0
        total_count = len(target_tickers)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(get_ghost_metrics, t, min_volume_threshold): t 
                for t in target_tickers
            }
            
            for future in as_completed(future_to_ticker):
                data = future.result()
                if data:
                    results.append(data)
                
                completed_count += 1
                progress_bar.progress(completed_count / total_count)
            
        status.update(label=f"掃描完成！共發現 {len(results)} 檔。", state="complete", expanded=False)

    if results:
        df_results = pd.DataFrame(results)
        
        # HV Rank 由低到高排序
        df_results = df_results.sort_values(by="HV Rank", ascending=True)
        
        st.success(f"🎯 發現 {len(df_results)} 檔優質標的！")
        
        column_config = {
            "HV Rank": st.column_config.NumberColumn("HV波動 (低優先)", format="%.1f"),
            "現價": st.column_config.NumberColumn(format="$%.2f"),
            "4H 60MA": st.column_config.NumberColumn("4H 季線", format="$%.2f"),
            "乖離率": st.column_config.TextColumn("距離均線"),
            "產業": st.column_config.TextColumn("產業 (中文)"),
            "財報日": st.column_config.TextColumn("下季財報"),
            # 【新功能】這裡會顯示一個連結按鈕
            "題材搜尋": st.column_config.LinkColumn(
                "題材與風險", 
                display_text="🔍 點擊查詢", # 按鈕上顯示的文字
                help="點擊後將跳轉至 Google 搜尋該股票的最新分析與題材"
            ),
            "_sort_score": None,
            "_dist_raw": None
        }

        if enable_u_logic:
            column_config["U型強度"] = st.column_config.ProgressColumn(
                "U型分數", 
                min_value=0, max_value=20, format="%.1f"
            )
        else:
             column_config["U型強度"] = st.column_config.NumberColumn("U型分數 (未啟用)", format="%.1f")

        st.dataframe(
            df_results,
            column_config=column_config,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("⚠️ 沒掃到符合條件的股票。\n建議：\n1. 放寬「HV Rank 門檻」\n2. 嘗試取消勾選「日線 60MA 向上」")
