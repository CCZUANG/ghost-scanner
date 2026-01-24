import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器 (Turbo急速版)", page_icon="👻", layout="wide")

st.title("👻 幽靈策略掃描器 (Turbo急速版)")
st.write("""
**策略目標**：多核心平行運算，極速尋找 **「4小時 60MA 完美 U 型反轉」**。
""")

# --- 2. 側邊欄：參數設定區 ---
st.sidebar.header("⚙️ 基礎篩選")
# 為了證明速度，預設數量直接拉高
scan_limit = st.sidebar.slider("1. 掃描數量 (前 N 大)", 50, 500, 300, help="開啟多核心後，300檔也能很快掃完")
hv_threshold = st.sidebar.slider("2. HV Rank 門檻", 10, 90, 60)
min_vol_m = st.sidebar.slider("3. 最小日均量 (百萬股)", 1, 20, 3) 
min_volume_threshold = min_vol_m * 1000000

st.sidebar.header("📈 4小時 U型戰法")
dist_threshold = st.sidebar.slider("🎯 距離 60MA 範圍 (%)", 0.0, 50.0, 8.0, step=0.5)
u_sensitivity = st.sidebar.slider("U型敏感度 (Lookback)", 20, 60, 30)
min_curvature = st.sidebar.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")

st.sidebar.markdown("---")
# 新增執行緒設定
max_workers = st.sidebar.slider("🚀 平行運算核心數", 1, 32, 16, help="數字越大跑越快，但設太大可能會被 Yahoo 擋 IP，建議 10-20")

# --- 3. 核心函數 ---

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = (
            "https://en.wikipedia.org/wiki/"
            "List_of_S%26P_500_companies"
        )
        response = requests.get(url, headers=headers)
        sp500_df = pd.read_html(StringIO(response.text))[0]
        tickers = sp500_df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except:
        return ['TSM', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'PLTR', 'LUNR', 'COIN', 'MSTR']

def analyze_u_shape(ma_series):
    try:
        y = ma_series.values
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs
        
        if a <= 0: return False, 0 # 倒U或直線
        
        vertex_x = -b / (2 * a)
        len_window = len(y)
        
        # 寬鬆一點的谷底判定，抓最近的趨勢
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
        
        # 優化：一次抓取 3 個月的 1h 數據，日線數據從這裡面 resample 出來即可
        # 這樣可以少發送一次網路請求，速度更快
        df_1h = stock.history(period="3mo", interval="1h")
        
        if len(df_1h) < 240: return None

        # 1. 計算日均量 (用 1h 數據合成日線來算)
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
        
        # 取出這段時間的 MA 數據進行擬合
        ma_segment = df_4h['MA60'].iloc[-u_sensitivity:]
        if ma_segment.isnull().values.any() or len(ma_segment) < u_sensitivity: return None
        
        # --- 核心演算法：U 型檢測 ---
        is_u_shape, curvature = analyze_u_shape(ma_segment)
        
        if not is_u_shape: return None
        if curvature < min_curvature: return None
        
        # 乖離率檢查
        current_price = df_4h['Close'].iloc[-1]
        ma60_now = ma_segment.iloc[-1]
        dist_pct = ((current_price - ma60_now) / ma60_now) * 100
        
        if abs(dist_pct) > dist_threshold: return None 

        # 計算排序分數
        u_score = (curvature * 1000) - (abs(dist_pct) * 0.5)

        # 計算 HV Rank (簡單版)
        # 用 4H 的波動率來估算
        log_ret = np.log(df_4h['Close'] / df_4h['Close'].shift(1))
        current_hv = log_ret.rolling(window=30).std().iloc[-1] * 100 * 2 # 粗略放大

        return {
            "代號": symbol,
            "現價": round(current_price, 2),
            "4H 60MA": round(ma60_now, 2),
            "U型強度": round(curvature * 1000, 2),
            "乖離率": f"{round(dist_pct, 2)}%",
            "狀態": "✅ 完美微笑",
            "_sort_score": u_score,
            "_dist_raw": abs(dist_pct)
        }
    except:
        return None

# --- 4. 主程式執行邏輯 (多執行緒版) ---

if st.button("🚀 啟動 Turbo 掃描", type="primary"):
    status_text = "正在下載 S&P 500 清單..."
    progress_bar = st.progress(0)
    
    with st.status(status_text, expanded=True) as status:
        tickers = get_sp500_tickers()
        target_tickers = tickers[:scan_limit]
        
        status.write(f"🔥 Turbo 模式啟動！ (核心數: {max_workers})")
        status.write(f"🔍 掃描中... 目標: {len(target_tickers)} 檔")
        
        results = []
        completed_count = 0
        total_count = len(target_tickers)
        
        # 使用 ThreadPoolExecutor 進行並行處理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務
            future_to_ticker = {
                executor.submit(get_ghost_metrics, t, min_volume_threshold): t 
                for t in target_tickers
            }
            
            # 當任務完成時處理結果
            for future in as_completed(future_to_ticker):
                data = future.result()
                if data:
                    results.append(data)
                
                completed_count += 1
                # 更新進度條
                progress_bar.progress(completed_count / total_count)
            
        status.update(label=f"掃描完成！耗時極短，共掃描 {total_count} 檔。", state="complete", expanded=False)

    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="U型強度", ascending=False)
        
        st.success(f"🎯 發現 {len(df_results)} 檔 U 型潛力股！")
        
        st.dataframe(
            df_results,
            column_config={
                "U型強度": st.column_config.ProgressColumn(
                    "U型分數", 
                    help="數值越高越彎，型態越漂亮",
                    min_value=0, max_value=20, format="%.1f"
                ),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "4H 60MA": st.column_config.NumberColumn(format="$%.2f"),
                "乖離率": st.column_config.TextColumn("距離均線"),
                "狀態": st.column_config.TextColumn("型態"),
                "_sort_score": None,
                "_dist_raw": None
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("⚠️ 沒掃到符合條件的股票。\n建議：\n1. 降低「最小彎曲度」\n2. 擴大「距離 60MA 範圍」")
