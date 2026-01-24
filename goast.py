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
        # 備用清單
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
        
        # 2. 波動 (HV Rank) --- 這裡是之前報錯的地方，已修復 ---
        log_ret = np.log(close / close.shift(1))
        
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        current_hv = vol_30d.iloc[-1]
        min_hv = vol_30d.min()
        max_hv = vol_30d.max()
        
        if max_hv == min_hv: return None
        hv_rank = ((current_hv - min_hv) / (max_hv - min_hv)) * 100
        
        # --- C. 型態判別 (Pattern Recognition) ---
        pattern = "📈 穩健上漲" # 預設值
        
        # 計算布林通道
        std20 = close.rolling(window=20).std().iloc[-1]
        upper_band = sma20 + (2 * std20)
        lower_band = sma20 - (2 * std20)
        
        # 指標 1: 布林帶寬 (Bandwidth)
        bb_width = (upper_band - lower_band) / sma20
        
        # 指標 2: 乖離率 (Bias)
        bias_pct = (current_price - sma20) / sma20
        
        # --- 判斷邏輯 ---
        
        # 1. 判斷【極度壓縮】
        if bb_width < 0.15:
            pattern = "🧊 極度壓縮 (關注!)"
            
        # 2. 判斷【回測支撐】
        elif 0 < bias_pct < 0.02:
            pattern = "📉 回測支撐 (買點)"
            
        # 3. 判斷【強勢突破】
        elif current_price > upper_band:
            pattern = "🚀 強勢突破 (慎追)"

        return {
            "代號": symbol,
            "現價": round(current_price, 2),
            "HV Rank": round(hv_rank, 1),
            "趨勢": "✅" if trend_up else "❌",
            "型態特徵": pattern,
            "日均量": f"{round(avg_volume/1000000, 1)}M"
        }
    except:
        return None

# --- 4. 主程式執行邏輯 ---

if st.button("🚀 開始掃描", type="primary"):
    status_text = "正在下載 S&P 500 清單..."
    progress_bar = st.progress(0)
    
    with st.status(status_text, expanded=True) as status:
        tickers = get_sp500_tickers()
        target_tickers = tickers[:scan_limit]
        
        status.write(f"🔍 掃描中... (條件：日均量 > {min_vol_m}M 且 HV Rank < {hv_threshold})")
        
        results = []
        for i, ticker in enumerate(target_tickers):
            data = get_ghost_metrics(ticker, min_volume_threshold)
            
            # 篩選
            if data and data['趨勢'] == "✅" and data['HV Rank'] < hv_threshold:
                results.append(data)
            
            progress_bar.progress((i + 1) / len(target_tickers))
            
        status.update(label="掃描完成！", state="complete", expanded=False)

    # --- 5. 顯示結果 ---
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="HV Rank")
        
        st.success(f"🎯 發現 {len(df_results)} 檔標的！請特別關注標註「🧊」或「📉」的股票。")
        
        st.dataframe(
            df_results,
            column_config={
                "HV Rank": st.column_config.NumberColumn("波動位階", format="%.1f"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "型態特徵": st.column_config.TextColumn("K線型態 (重點)", help="🧊=壓縮準備噴發, 📉=回檔低接"),
                "日均量": st.column_config.TextColumn("成交量")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("無符合條件標的，請放寬 HV Rank 門檻或降低日均量。")
