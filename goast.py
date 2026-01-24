import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO

# --- 頁面設定 ---
st.set_page_config(page_title="幽靈策略掃描器", page_icon="👻")

st.title("👻 幽靈策略掃描器")
st.write("專為《華爾街幽靈》策略設計，尋找 **低波動 (HV Rank < 30)** 且 **趨勢向上** 的 S&P 500 標的。")

# --- 側邊欄設定 ---
st.sidebar.header("設定參數")
scan_limit = st.sidebar.slider("掃描數量 (前 N 大)", 50, 500, 100)
hv_threshold = st.sidebar.slider("HV Rank 門檻 (低於多少)", 10, 50, 30)

# --- 核心函數 ---
@st.cache_data(ttl=3600) # 快取 1 小時，避免重複抓清單
def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, headers=headers)
        sp500_df = pd.read_html(StringIO(response.text))[0]
        tickers = sp500_df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except:
        return ['TSM', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD']

def get_ghost_metrics(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        if len(df) < 100: return None
        
        current_price = df['Close'].iloc[-1]
        sma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        trend_up = current_price > sma20
        
        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        
        current_hv = vol_30d.iloc[-1]
        min_hv = vol_30d.min()
        max_hv = vol_30d.max()
        
        if max_hv == min_hv: return None
        hv_rank = ((current_hv - min_hv) / (max_hv - min_hv)) * 100
        
        return {
            "代號": symbol,
            "現價": round(current_price, 2),
            "HV Rank": round(hv_rank, 1),
            "趨勢": "✅" if trend_up else "❌",
            "狀態": "🥶 適合 Step 1" if (trend_up and hv_rank < 30) else "觀察中"
        }
    except:
        return None

# --- 主程式邏輯 ---
if st.button("🚀 開始掃描"):
    with st.status("正在下載 S&P 500 清單...", expanded=True) as status:
        tickers = get_sp500_tickers()
        status.write(f"✅ 取得 {len(tickers)} 檔股票，開始分析前 {scan_limit} 檔...")
        
        results = []
        progress_bar = st.progress(0)
        
        # 掃描迴圈
        for i, ticker in enumerate(tickers[:scan_limit]):
            data = get_ghost_metrics(ticker)
            # 根據使用者設定的門檻過濾
            if data and data['趨勢'] == "✅" and data['HV Rank'] < hv_threshold:
                results.append(data)
            
            # 更新進度條
            progress_bar.progress((i + 1) / scan_limit)
            
        status.update(label="掃描完成！", state="complete", expanded=False)

    # --- 顯示結果 ---
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="HV Rank")
        
        st.success(f"🎯 發現 {len(df_results)} 檔符合條件的標的！")
        st.dataframe(
            df_results,
            column_config={
                "HV Rank": st.column_config.NumberColumn(
                    "波動位階 (越低越好)",
                    help="0=年度最低波动, 100=年度最高波动",
                    format="%.1f %%"
                ),
                "現價": st.column_config.NumberColumn(format="$%.2f")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("沒有發現符合條件的股票，請嘗試放寬 HV 門檻。")