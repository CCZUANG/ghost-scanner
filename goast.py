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
min_vol_m = st.sidebar.slider("3. 最小日均量 (百萬股)", 1, 20, 3) # 預設調低一點以免濾掉太多
min_volume_threshold = min_vol_m * 1000000

st.sidebar.header("📈 4小時 60MA 戰法")
only_ma_flip = st.sidebar.checkbox("✅ 嚴格篩選「微笑轉折」", value=True, help="只顯示 MA60 呈現 U 型反轉 (左跌右漲) 的股票")
dist_threshold = st.sidebar.slider("🎯 距離 60MA 範圍 (%)", 0.0, 10.0, 3.0, step=0.5, help="股價距離 60MA 多近？")

st.sidebar.markdown("---")
st.sidebar.info("💡 **圖形辨識邏輯**：\n程式會檢查過去 5 根 4H K棒的均線走勢，尋找「先跌、後平、再勾起」的 U 型結構。")

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
        return ['TSM', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'PLTR', 'LUNR', 'COIN']

def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol)
        
        # --- 第一階段：日線快篩 ---
        df_daily = stock.history(period="6mo")
        if len(df_daily) < 100: return None
        
        # 1. 流動性
        avg_volume = df_daily['Volume'].rolling(window=20).mean().iloc[-1]
        if avg_volume < vol_threshold: return None 
        
        # 2. 波動率 (HV Rank)
        close_daily = df_daily['Close']
        log_ret = np.log(close_daily / close_daily.shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        current_hv = vol_30d.iloc[-1]
        min_hv = vol_30d.min()
        max_hv = vol_30d.max()
        if max_hv == min_hv: return None
        hv_rank = ((current_hv - min_hv) / (max_hv - min_hv)) * 100
        
        if hv_rank > hv_threshold: return None

        # --- 第二階段：4小時 K線與 60MA 深度分析 ---
        # 抓取 1H 數據合成 4H
        df_1h = stock.history(period="3mo", interval="1h")
        if len(df_1h) < 240: return None

        # 合成 4H K線
        df_4h = df_1h.resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # 計算 4H 的 60MA (這是關鍵指標)
        df_4h['MA60'] = df_4h['Close'].rolling(window=60).mean()
        
        # 取出最後 10 根 MA60 數值來分析型態
        ma60_recent = df_4h['MA60'].iloc[-10:]
        if ma60_recent.isnull().values.any(): return None
        
        # 取得關鍵點位
        current_price = df_4h['Close'].iloc[-1]
        ma60_now = ma60_recent.iloc[-1]      # 現在的 MA
        ma60_prev = ma60_recent.iloc[-2]     # 1根前的 MA (4小時前)
        ma60_prev_3 = ma60_recent.iloc[-4]   # 3根前的 MA (12小時前)
        ma60_prev_5 = ma60_recent.iloc[-6]   # 5根前的 MA (20小時前)

        # --- A. 判斷是否為「微笑曲線」 (U-Shape Turn) ---
        # 邏輯 1: 現在必須是向上的 (末端上勾)
        is_rising_now = ma60_now > ma60_prev
        
        # 邏輯 2: 之前必須是向下的或平的 (確認它是從底部翻起來，而不是一直漲)
        # 我們檢查 5 根 K 棒前的 MA 是否比 2 根 K 棒前的高 (代表之前是跌勢)
        was_falling = ma60_prev_5 > ma60_prev_3 
        
        # 綜合判定: 剛翻揚 = 現在漲 + 之前跌/平
        is_smile_turn = is_rising_now and was_falling

        # --- B. 計算乖離率 ---
        dist_pct = ((current_price - ma60_now) / ma60_now) * 100
        
        # --- 篩選邏輯 ---
        
        # 篩選 1: 微笑轉折 (如果使用者有勾選)
        if only_ma_flip and not is_smile_turn: return None
        
        # 篩選 2: 乖離率 (距離 60MA 不能太遠)
        if abs(dist_pct) > dist_threshold: return None
        
        # 篩選 3: 價格必須在 MA60 之上 (支撐有效)
        if current_price < ma60_now: return None

        return {
            "代號": symbol,
            "現價": round(current_price, 2),
            "HV Rank": round(hv_rank, 1),
            "4H 60MA": round(ma60_now, 2),
            "狀態": "U型反轉 ✅" if is_smile_turn else "持續上漲 ↗️",
            "距離均線": f"{round(dist_pct, 2)}%",
            "_dist_raw": abs(dist_pct)
        }
    except Exception as e:
        return None

# --- 4. 主程式執行邏輯 ---

if st.button("🚀 啟動 4H 微笑掃描", type="primary"):
    status_text = "正在下載 S&P 500 清單..."
    progress_bar = st.progress(0)
    
    with st.status(status_text, expanded=True) as status:
        tickers = get_sp500_tickers()
        target_tickers = tickers[:scan_limit]
        
        status.write(f"🔍 掃描中... \n找尋「MA60 剛翻揚」且股價貼近支撐的股票")
        
        results = []
        for i, ticker in enumerate(target_tickers):
            data = get_ghost_metrics(ticker, min_volume_threshold)
            if data:
                results.append(data)
            progress_bar.progress((i + 1) / len(target_tickers))
            
        status.update(label="分析完成！", state="complete", expanded=False)

    # --- 5. 顯示結果 ---
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="_dist_raw")
        
        st.success(f"🎯 發現 {len(df_results)} 檔「4H 60MA 剛啟動」的股票！")
        
        st.dataframe(
            df_results,
            column_config={
                "HV Rank": st.column_config.NumberColumn("波動位階", format="%.1f"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "4H 60MA": st.column_config.NumberColumn(format="$%.2f", help="4小時線的季線位置"),
                "狀態": st.column_config.TextColumn("型態判定"),
                "距離均線": st.column_config.TextColumn("乖離率", help="越小代表買點越漂亮 (剛回測完)"),
                "_dist_raw": None
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("⚠️ 沒掃到符合「微笑曲線」的股票。\n\n這代表目前大多數股票可能已經漲了一段時間，或者趨勢不明。\n建議：取消勾選「嚴格篩選」，看看那些已經是持續上漲趨勢的股票。")
