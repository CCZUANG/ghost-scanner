import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器 (趨勢防護版)", page_icon="👻", layout="wide")

st.title("👻 幽靈策略掃描器 (趨勢防護版)")
st.write("""
**策略目標**：在 **日線多頭 (季線向上)** 的保護下，尋找 **4小時 U 型起漲點**。
""")

# --- 2. 側邊欄：參數設定區 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio(
    "選擇掃描市場", 
    ["S&P 500 (大型股)", "NASDAQ 100 (科技股)", "🔥 全火力 (兩者全掃)"],
    index=2
)
scan_limit = st.sidebar.slider("掃描數量 (前 N 大)", 50, 600, 200)

# --- 新增：日線趨勢濾網 (User 指定功能) ---
st.sidebar.header("🛡️ 日線趨勢濾網 (避免接刀)")
check_daily_ma60_up = st.sidebar.checkbox("✅ 必須：日線 60MA 向上 (排除空頭)", value=True, help="強制過濾掉日線季線還在下彎的股票，避免買到空頭反彈。")
check_price_above_daily_ma60 = st.sidebar.checkbox("✅ 必須：股價 > 日線 60MA", value=True, help="確保股價站上季線，屬於多頭格局。")

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
        # 抓取 6 個月數據 (確保日線 MA60 有足夠資料)
        df_1h = stock.history(period="6mo", interval="1h")
        
        if len(df_1h) < 240: return None

        # --- A. 日線級別處理 (Daily Analysis) ---
        df_daily_synth = df_1h.resample('D').agg({
            'Volume': 'sum',
            'Close': 'last'
        }).dropna()
        
        # 1. 計算日線 MA60
        df_daily_synth['MA60'] = df_daily_synth['Close'].rolling(window=60).mean()
        
        # 檢查資料長度
        if len(df_daily_synth) < 60: return None
        
        daily_ma60_now = df_daily_synth['MA60'].iloc[-1]
        daily_ma60_prev = df_daily_synth['MA60'].iloc[-2]
        current_price_daily = df_daily_synth['Close'].iloc[-1]

        # === 🛡️ 日線趨勢過濾 (關鍵修改) ===
        
        # 條件 1: 日線 MA60 必須向上 (斜率 > 0)
        if check_daily_ma60_up:
            if daily_ma60_now <= daily_ma60_prev: 
                return None # 季線還在跌，淘汰
        
        # 條件 2: 股價必須在 日線 MA60 之上
        if check_price_above_daily_ma60:
            if current_price_daily < daily_ma60_now:
                return None # 股價在季線下，淘汰

        # 2. 成交量過濾
        avg_volume = df_daily_synth['Volume'].rolling(window=20).mean().iloc[-1]
        if avg_volume < vol_threshold: return None

        # 3. HV Rank 過濾
        close_daily = df_daily_synth['Close']
        log_ret = np.log(close_daily / close_daily.shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        
        current_hv = vol_30d.iloc[-1]
        min_hv = vol_30d.min()
        max_hv = vol_30d.max()
        
        if max_hv == min_hv: return None
        hv_rank = ((current_hv - min_hv) / (max_hv - min_hv)) * 100
        
        if hv_rank > hv_threshold: return None

        # --- B. 4小時級別處理 (4H Analysis) ---
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

        # 乖離率過濾
        if abs(dist_pct) > dist_threshold: return None 
        
        # --- C. U 型檢測邏輯 ---
        u_score = 0
        curvature = 0
        status_msg = "符合乖離"

        if enable_u_logic:
            is_u_shape, curv = analyze_u_shape(ma_segment)
            if not is_u_shape: return None
            if curv < min_curvature: return None
            
            curvature = curv
            status_msg = "✅ 完美微笑"
            u_score = (curvature * 1000) - (abs(dist_pct) * 0.5)
        else:
            u_score = -abs(dist_pct)
            curvature = 0 

        # --- D. 期權檢查 ---
        try:
            if not stock.options: 
                return None
        except:
            return None

        return {
            "代號": symbol,
            "HV Rank": round(hv_rank, 1),
            "現價": round(current_price_4h, 2),
            "4H 60MA": round(ma60_now_4h, 2),
            "日線 60MA": "⬆️ 翻揚" if daily_ma60_now > daily_ma60_prev else "⬇️ 下彎",
            "U型強度": round(curvature * 1000, 2),
            "乖離率": f"{round(dist_pct, 2)}%",
            "狀態": status_msg,
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
        
        # 顯示過濾條件資訊
        msg_trend = "已啟用" if check_daily_ma60_up else "未啟用"
        msg_u = "已啟用" if enable_u_logic else "未啟用"
        
        status.write(f"🛡️ 日線趨勢過濾: {msg_trend} | U型過濾: {msg_u}")
        
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
        df_results = df_results.sort_values(by="_sort_score", ascending=False)
        
        st.success(f"🎯 發現 {len(df_results)} 檔優質標的！")
        
        column_config = {
            "HV Rank": st.column_config.NumberColumn("HV波動", format="%.1f"),
            "現價": st.column_config.NumberColumn(format="$%.2f"),
            "4H 60MA": st.column_config.NumberColumn("4H 季線", format="$%.2f"),
            "日線 60MA": st.column_config.TextColumn("日線趨勢"),
            "乖離率": st.column_config.TextColumn("距離均線"),
            "狀態": st.column_config.TextColumn("型態"),
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
        st.warning("⚠️ 沒掃到符合條件的股票。\n可能原因：\n1. 「日線趨勢」條件太嚴格（現在是修正期？）\n2. 嘗試取消勾選「日線 60MA 向上」看看有無反彈股。")
