import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器 (U型究極版)", page_icon="👻", layout="wide")

st.title("👻 幽靈策略掃描器 (U型究極版)")
st.write("""
**策略目標**：利用數學擬合演算法，精準捕捉 **「4小時 60MA 完美 U 型反轉」** 的標的。
""")

# --- 2. 側邊欄：參數設定區 ---
st.sidebar.header("⚙️ 基礎篩選")
scan_limit = st.sidebar.slider("1. 掃描數量 (前 N 大)", 50, 500, 150)
hv_threshold = st.sidebar.slider("2. HV Rank 門檻", 10, 90, 60, help="找反轉型態時，波動率可以設寬一點")
min_vol_m = st.sidebar.slider("3. 最小日均量 (百萬股)", 1, 20, 3) 
min_volume_threshold = min_vol_m * 1000000

st.sidebar.header("📈 4小時 60MA 戰法")
dist_threshold = st.sidebar.slider("🎯 距離 60MA 範圍 (%)", 0.0, 50.0, 10.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🧠 U型演算法設定")
u_sensitivity = st.sidebar.slider("U型敏感度 (Lookback)", 20, 60, 30, help="要看過去幾根 K 棒來畫 U 型？(30根約等於5天)")
min_curvature = st.sidebar.slider("最小彎曲度 (Curvature)", 0.0, 0.1, 0.005, format="%.3f", help="數值越高，U 型越深、越明顯；數值越低越平緩。")

st.sidebar.info("💡 **數學原理**：\n程式會對 MA60 進行二次微分擬合，計算出拋物線係數。只有符合「開口向上」且「谷底剛過」的股票才會被選出。")

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
    """
    使用二次多項式擬合 (Quadratic Regression) 來判斷 U 型
    y = ax^2 + bx + c
    a > 0 代表開口向上 (U型)
    頂點位置 x = -b / (2a) 代表谷底發生的時間點
    """
    try:
        y = ma_series.values
        x = np.arange(len(y))
        
        # 進行二次擬合
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs
        
        # 1. 檢查開口方向 (a 必須大於 0 才是 U 型，小於 0 是倒 U)
        if a <= 0: return False, 0, "倒U或直線"
        
        # 2. 計算谷底位置 (Vertex)
        vertex_x = -b / (2 * a)
        
        # 3. 判斷谷底位置是否合理
        # 谷底必須發生在觀察期間的「中後段」，但不能是「未來」(> len) 或 「太久以前」 (< 0)
        # 我們希望谷底剛剛發生 (例如在最後 30% ~ 90% 的區間)
        len_window = len(y)
        if not (len_window * 0.4 <= vertex_x <= len_window * 1.0):
            return False, a, "谷底位置不對"
            
        # 4. 檢查現在的斜率 (確保右邊是翹起來的)
        current_slope = y[-1] - y[-2]
        if current_slope <= 0: return False, a, "右側未勾起"

        return True, a, "完美U型"
    except:
        return False, 0, "計算錯誤"

def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol)
        
        # --- 第一階段：日線快篩 ---
        df_daily = stock.history(period="6mo")
        if len(df_daily) < 100: return None
        
        avg_volume = df_daily['Volume'].rolling(window=20).mean().iloc[-1]
        if avg_volume < vol_threshold: return None 
        
        close_daily = df_daily['Close']
        log_ret = np.log(close_daily / close_daily.shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        hv_rank = vol_30d.iloc[-1] # 簡化直接用數值比較

        # --- 第二階段：4小時 K線深度分析 ---
        # 抓取更多數據以確保 MA60 穩定
        df_1h = stock.history(period="3mo", interval="1h")
        if len(df_1h) < 240: return None

        # 合成 4H K線
        df_4h = df_1h.resample('4h').agg({
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        df_4h['MA60'] = df_4h['Close'].rolling(window=60).mean()
        
        # 取出這段時間的 MA 數據進行擬合 (根據側邊欄設定的長度)
        ma_segment = df_4h['MA60'].iloc[-u_sensitivity:]
        if ma_segment.isnull().values.any() or len(ma_segment) < u_sensitivity: return None
        
        # --- 核心演算法：U 型檢測 ---
        is_u_shape, curvature, note = analyze_u_shape(ma_segment)
        
        # 取得關鍵數據
        current_price = df_4h['Close'].iloc[-1]
        ma60_now = ma_segment.iloc[-1]
        dist_pct = ((current_price - ma60_now) / ma60_now) * 100
        
        # --- 篩選邏輯 ---
        if not is_u_shape: return None
        if curvature < min_curvature: return None # 過濾掉太扁平的 U
        if abs(dist_pct) > dist_threshold: return None # 乖離率過濾
        
        # 計算分數 (Curvature 越大越好，且距離均線越近越好)
        # 這是一個自定義分數，用來排序
        u_score = (curvature * 1000) - (abs(dist_pct) * 0.5)

        return {
            "代號": symbol,
            "現價": round(current_price, 2),
            "4H 60MA": round(ma60_now, 2),
            "U型強度": round(curvature * 1000, 2), # 放大顯示方便閱讀
            "乖離率": f"{round(dist_pct, 2)}%",
            "狀態": "✅ 完美微笑",
            "_sort_score": u_score, # 排序用
            "_dist_raw": abs(dist_pct)
        }
    except:
        return None

# --- 4. 主程式執行邏輯 ---

if st.button("🚀 啟動 U型 數學擬合掃描", type="primary"):
    status_text = "正在下載 S&P 500 清單..."
    progress_bar = st.progress(0)
    
    with st.status(status_text, expanded=True) as status:
        tickers = get_sp500_tickers()
        target_tickers = tickers[:scan_limit]
        
        status.write(f"🔍 掃描中... \n演算法：二次微分擬合 (Lookback={u_sensitivity})")
        
        results = []
        for i, ticker in enumerate(target_tickers):
            data = get_ghost_metrics(ticker, min_volume_threshold)
            if data:
                results.append(data)
