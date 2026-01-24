import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器", page_icon="👻")

st.title("👻 幽靈策略掃描器")
st.write("""
專為《華爾街幽靈》策略設計的即時掃描工具。
尋找 **低波動 (結冰)**、**趨勢向上** 且 **流動性充足** 的 S&P 500 標的。
""")

# --- 2. 側邊欄：參數設定區 ---
st.sidebar.header("⚙️ 參數設定")

# 參數 A: 掃描範圍
scan_limit = st.sidebar.slider("1. 掃描數量 (前 N 大)", 50, 500, 100, help="為了手機速度，建議設 100 左右")

# 參數 B: 波動率門檻 (HV Rank)
hv_threshold = st.sidebar.slider("2. HV Rank 門檻 (低於多少)", 10, 60, 30, help="越低代表越便宜 (水結冰)，通常 <30 適合 Step 1")

# 參數 C: 流動性門檻 (新增功能！)
min_vol_m = st.sidebar.slider("3. 最小日均量 (百萬股)", 1, 20, 5, help="過濾掉沒人玩的死魚股。建議至少 5M 以確保期權好進出。")
min_volume_threshold = min_vol_m * 1000000  # 換算成實際股數

st.sidebar.markdown("---")
st.sidebar.info("💡 **提示**：\n數值越低越嚴格，找到的股票越少，但質量越高。")

# --- 3. 核心函數：抓清單與計算指標 ---

@st.cache_data(ttl=3600) # 快取 1 小時，避免重複抓清單浪費時間
def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0"}
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, headers=headers)
        sp500_df = pd.read_html(StringIO(response.text))[0]
        tickers = sp500_df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers] # 修正 BRK.B
        return tickers
    except Exception as e:
        # 如果爬蟲失敗，回傳備用的熱門股清單
        return ['TSM', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 
               'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'LUNR', 'PLTR', 'COIN', 'MSTR', 'SMCI']

def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol)
        # 抓取 6 個月數據 (兼顧速度與計算需求)
        df = stock.history(period="6mo")
        
        if len(df) < 100: return None
        
        # --- A. 流動性過濾 (Liquidity Check) ---
        # 計算過去 20 天平均成交量
        avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
        
        # 如果成交量小於使用者設定的門檻，直接跳過 (節省運算)
        if avg_volume < vol_threshold: return None 
        
        # --- B. 技術指標計算 ---
        current_price = df['Close'].iloc[-1]
        
        # 1. 趨勢判定 (站上 20MA)
        sma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        trend_up = current_price > sma20
        
        # 2. 波動率位階 (HV Rank)
        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        # 年化歷史波動率 (30天)
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        
        current_hv = vol_30d.iloc[-1]
        min_hv = vol_30d.min()
        max_hv = vol_30d.max()
        
        # 避免分母為 0
        if max_hv == min_hv: return None
        
        hv_rank = ((current_hv - min_hv) / (max_hv - min_hv)) * 100
        
        return {
            "代號": symbol,
            "現價": round(current_price, 2),
            "HV Rank": round(hv_rank, 1),
            "趨勢": "✅" if trend_up else "❌",
            # 將成交量格式化為百萬 (M)
            "日均量": f"{round(avg_volume/1000000, 1)}M",
            # 原始數值用於排序，之後會隱藏
            "_vol_raw": avg_volume 
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
        
        status.write(f"✅ 成功取得清單，開始分析前 {len(target_tickers)} 檔股票...")
        status.write(f"🔍 過濾條件：日均量 > {min_vol_m}M 且 HV Rank < {hv_threshold}")
        
        results = []
        
        # 迴圈掃描
        for i, ticker in enumerate(target_tickers):
            # 傳入使用者設定的 vol_threshold
            data = get_ghost_metrics(ticker, min_volume_threshold)
            
            # 根據 HV Rank 與 趨勢 進行最後篩選
            if data and data['趨勢'] == "✅" and data['HV Rank'] < hv_threshold:
                results.append(data)
            
            # 更新進度條
            progress_bar.progress((i + 1) / len(target_tickers))
            
        status.update(label="掃描完成！", state="complete", expanded=False)

    # --- 5. 顯示結果表格 ---
    if results:
        df_results = pd.DataFrame(results)
        
        # 依照 HV Rank 由低到高排序 (越低代表越適合 Step 1)
        df_results = df_results.sort_values(by="HV Rank")
        
        st.success(f"🎯 共發現 {len(df_results)} 檔優質標的！")
        
        st.dataframe(
            df_results,
            column_config={
                "代號": st.column_config.TextColumn("股票代號"),
                "HV Rank": st.column_config.NumberColumn(
                    "波動位階 (越低越好)",
                    help="0=年度最低波動 (冰), 100=年度最高波動 (火)",
                    format="%.1f"
                ),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "日均量": st.column_config.TextColumn("日均量 (20日)"),
                "趨勢": st.column_config.TextColumn("多頭排列"),
                "_vol_raw": None # 隱藏這個欄位，不顯示給使用者看
            },
            hide_index=True,
            use_container_width=True
        )
        st.markdown("*註：若清單為空，請嘗試降低日均量要求，或調高 HV Rank 門檻。*")
        
    else:
        st.warning(f"😔 在前 {scan_limit} 檔股票中，沒有發現符合條件的標的。\n\n建議：\n1. 調高 HV Rank 門檻 (例如 40)\n2. 降低日均量要求\n3. 擴大掃描數量")
