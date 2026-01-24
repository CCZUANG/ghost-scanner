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
**策略目標**：尋找「低波動 + 日線趨勢向上」且 **「4小時 60MA 支撐轉強」** 的標的。
""")

# --- 2. 側邊欄：參數設定區 ---
st.sidebar.header("⚙️ 基礎篩選")
scan_limit = st.sidebar.slider("1. 掃描數量 (前 N 大)", 50, 500, 100)
hv_threshold = st.sidebar.slider("2. HV Rank 門檻 (低於多少)", 10, 60, 45, help="先用寬鬆一點的標準，再用4H均線過濾")
min_vol_m = st.sidebar.slider("3. 最小日均量 (百萬股)", 1, 20, 5)
min_volume_threshold = min_vol_m * 1000000

st.sidebar.header("📈 4小時 K線特搜 (新功能)")
only_ma_flip = st.sidebar.checkbox("✅ 只選「60MA 剛翻揚」", value=False, help="嚴格篩選：MA60 前一根是平或跌，現在剛轉漲")
dist_threshold = st.sidebar.slider("🎯 距離 60MA 範圍 (%)", 0.0, 10.0, 2.5, step=0.5, help="股價距離 60MA 多近才算及格？(越小越貼近支撐)")

st.sidebar.markdown("---")
st.sidebar.info("💡 **4H 60MA 戰法**：\n4小時層級的 60MA 是波段生命線。當它翻揚且股價回測不破時，是勝率最高的 Step 1 進場點。")

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
        return ['TSM', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'PLTR', 'LUNR']

def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol)
        
        # --- 第一階段：日線快篩 (Daily Check) ---
        # 先抓日線，不合格的直接踢掉，節省抓取 1H 數據的時間
        df_daily = stock.history(period="6mo")
        if len(df_daily) < 100: return None
        
        # 1. 流動性過濾
        avg_volume = df_daily['Volume'].rolling(window=20).mean().iloc[-1]
        if avg_volume < vol_threshold: return None 
        
        # 2. 波動率過濾 (HV Rank)
        close_daily = df_daily['Close']
        log_ret = np.log(close_daily / close_daily.shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        current_hv = vol_30d.iloc[-1]
        min_hv = vol_30d.min()
        max_hv = vol_30d.max()
        
        if max_hv == min_hv: return None
        hv_rank = ((current_hv - min_hv) / (max_hv - min_hv)) * 100
        
        # 如果 HV 太高，直接淘汰，不需要跑第二階段
        if hv_rank > hv_threshold: return None

        # --- 第二階段：4小時 K線分析 (4H Analysis) ---
        # 抓取 1H 數據 (約 3 個月) 來合成 4H
        df_1h = stock.history(period="3mo", interval="1h")
        if len(df_1h) < 240: return None # 數據不足算 60MA

        # 合成 4H K線 (Resample)
        # 邏輯：每 4 條 1H 棒合成 1 條 4H 棒
        df_4h = df_1h.resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # 計算 4H 的 60MA
        df_4h['MA60'] = df_4h['Close'].rolling(window=60).mean()
        
        # 取得最新一根與前一根的數據
        current_4h_close = df_4h['Close'].iloc[-1]
        ma60_now = df_4h['MA60'].iloc[-1]
        ma60_prev = df_4h['MA60'].iloc[-2]
        
        if pd.isna(ma60_now) or pd.isna(ma60_prev): return None

        # A. 判斷 60MA 趨勢 (斜率)
        ma_slope = ma60_now - ma60_prev
        if ma_slope > 0.05: trend_4h = "↗️ 向上"
        elif ma_slope < -0.05: trend_4h = "↘️ 向下"
        else: trend_4h = "➡️ 持平"
        
        # B. 判斷是否「剛翻揚」 (Flip Up)
        # 前一根是平或跌，現在這根是漲
        is_flipping_up = (ma60_now > ma60_prev) and (df_4h['MA60'].iloc[-2] <= df_4h['MA60'].iloc[-3])

        # C. 計算乖離率 (Distance to MA)
        # 正數 = 股價在 MA 上方 n%
        dist_pct = ((current_4h_close - ma60_now) / ma60_now) * 100
        
        # --- 整合與篩選 ---
        
        # 篩選 1: 如果使用者勾選「剛翻揚」，則必須符合 flip 條件
        if only_ma_flip and not is_flipping_up: return None
        
        # 篩選 2: 檢查乖離率是否在使用者設定的範圍內 (取絕對值)
        if abs(dist_pct) > dist_threshold: return None

        # 篩選 3: 確保股價至少在 4H 60MA 之上 (支撐)
        # (如果您想找跌破翻空的可以拿掉這行，但幽靈策略 Step 1 做多為主)
        if current_4h_close < ma60_now * 0.99: return None # 容許跌破 1% 的假跌破

        return {
            "代號": symbol,
            "現價": round(current_4h_close, 2),
            "HV Rank": round(hv_rank, 1),
            "4H 60MA": round(ma60_now, 2),
            "均線方向": trend_4h,
            "剛翻揚?": "✅ YES" if is_flipping_up else "",
            "距離均線": f"{round(dist_pct, 2)}%",
            "_dist_raw": abs(dist_pct) # 排序用
        }
    except Exception as e:
        return None

# --- 4. 主程式執行邏輯 ---

if st.button("🚀 啟動 4H 掃描", type="primary"):
    status_text = "正在下載 S&P 500 清單..."
    progress_bar = st.progress(0)
    
    with st.status(status_text, expanded=True) as status:
        tickers = get_sp500_tickers()
        target_tickers = tickers[:scan_limit]
        
        status.write(f"🔍 掃描中... \n條件：日均量 > {min_vol_m}M \n乖離率 < {dist_threshold}% \n只選剛翻揚: {only_ma_flip}")
        
        results = []
        for i, ticker in enumerate(target_tickers):
            data = get_ghost_metrics(ticker, min_volume_threshold)
            if data:
                results.append(data)
            progress_bar.progress((i + 1) / len(target_tickers))
            
        status.update(label="4H 結構分析完成！", state="complete", expanded=False)

    # --- 5. 顯示結果 ---
    if results:
        df_results = pd.DataFrame(results)
        # 依照「距離均線」排序 (越貼近均線越好，代表停損空間小)
        df_results = df_results.sort_values(by="_dist_raw")
        
        st.success(f"🎯 發現 {len(df_results)} 檔符合 4H 架構的股票！")
        
        st.dataframe(
            df_results,
            column_config={
                "HV Rank": st.column_config.NumberColumn("波動位階", format="%.1f"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "4H 60MA": st.column_config.NumberColumn(format="$%.2f", help="4小時線的季線位置"),
                "均線方向": st.column_config.TextColumn("60MA 趨勢"),
                "剛翻揚?": st.column_config.TextColumn("轉強訊號"),
                "距離均線": st.column_config.TextColumn("乖離率", help="正數代表股價在均線上方"),
                "_dist_raw": None
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("⚠️ 沒有股票符合目前的 4H 條件。\n\n建議：\n1. 放寬「距離 60MA 範圍」\n2. 取消勾選「只選剛翻揚」\n3. 增加掃描數量")
