import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器", page_icon="👻", layout="wide")

# --- 2. 智慧狀態管理 (含還原邏輯) ---
if 'scan_limit' not in st.session_state: st.session_state.scan_limit = 600 
if 'min_vol_m' not in st.session_state: st.session_state.min_vol_m = 10
if 'dist_threshold' not in st.session_state: st.session_state.dist_threshold = 8.0
if 'u_sensitivity' not in st.session_state: st.session_state.u_sensitivity = 30

if 'backup' not in st.session_state:
    st.session_state.backup = {
        'scan_limit': 600, 'min_vol_m': 10, 'dist_threshold': 8.0, 'u_sensitivity': 30
    }

def handle_u_logic_toggle():
    if st.session_state.u_logic_key:
        st.session_state.backup.update({
            'scan_limit': st.session_state.scan_limit,
            'min_vol_m': st.session_state.min_vol_m,
            'dist_threshold': st.session_state.dist_threshold,
            'u_sensitivity': st.session_state.u_sensitivity
        })
        st.session_state.scan_limit = 600
        st.session_state.min_vol_m = 1
        st.session_state.dist_threshold = 50.0
        st.session_state.u_sensitivity = 60
    else:
        st.session_state.scan_limit = st.session_state.backup['scan_limit']
        st.session_state.min_vol_m = st.session_state.backup['min_vol_m']
        st.session_state.dist_threshold = st.session_state.backup['dist_threshold']
        st.session_state.u_sensitivity = st.session_state.backup['u_sensitivity']

st.title("👻 幽靈策略掃描器")

# --- 3. 核心策略導引區 ---
with st.expander("📖 幽靈策略：動態蝴蝶演化步驟", expanded=True):
    col_step1, col_step2, col_step3 = st.columns(3)
    with col_step1:
        st.markdown("### Step 1：建立試探 (Rule 1)")
        st.markdown("買進 Low Call + 賣出 High Call (**多頭價差**)。")
    with col_step2:
        st.markdown("### Step 2：動能加碼 (Rule 2)")
        st.markdown("價差浮盈且衝向賣出價位時，**加買高階 Call**。")
    with col_step3:
        st.markdown("### Step 3：轉化蝴蝶")
        st.markdown("再加賣一張中間價位 Call，達成 **負成本穩定獲利**。")
    st.info("💡 **核心提醒**：Step 2 重點在於 **IV 擴張（水結成冰）**。")

st.markdown("---")

# --- 4. 側邊欄 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio("市場", ["S&P 500", "NASDAQ 100", "🔥 全火力"], index=2)

st.sidebar.header("📈 戰法連動")
enable_u_logic = st.sidebar.checkbox("✅ 啟動 4小時 U型戰法連動", value=False, key='u_logic_key', on_change=handle_u_logic_toggle)
scan_limit = st.sidebar.slider("掃描數量", 50, 600, key='scan_limit')

st.sidebar.header("🛡️ 趨勢濾網")
check_daily_ma60_up = st.sidebar.checkbox("✅ 日線 60MA 向上", value=True)
check_price_above_daily_ma60 = st.sidebar.checkbox("✅ 股價 > 日線 60MA", value=True)

st.sidebar.header("⚙️ 基礎篩選")
hv_threshold = st.sidebar.slider("HV Rank 門檻 (越低越好)", 10, 100, 30)
min_vol_m = st.sidebar.slider("最小日均量 (M)", 1, 100, key='min_vol_m') 
dist_threshold = st.sidebar.slider("距離 4H 60MA 範圍 (%)", 0.0, 50.0, key='dist_threshold', step=0.5)

if enable_u_logic:
    u_sensitivity = st.sidebar.slider("U型敏感度", 20, 60, key='u_sensitivity')
    min_curvature = st.sidebar.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")
else:
    u_sensitivity, min_curvature = 30, 0.003

max_workers = st.sidebar.slider("🚀 平行運算核心數", 1, 32, 16)

# --- 5. 核心運算函數 (【強烈修正】Wikipedia 抓取邏輯) ---

@st.cache_data(ttl=3600)
def get_tickers_robust(choice):
    headers = {"User-Agent": "Mozilla/5.0"}
    all_tickers = []
    
    # 抓取 S&P 500
    if choice in ["S&P 500", "🔥 全火力"]:
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[0]
            col = [c for c in df.columns if 'Symbol' in c or 'Ticker' in c][0]
            all_tickers.extend(df[col].tolist())
        except: pass

    # 抓取 NASDAQ 100
    if choice in ["NASDAQ 100", "🔥 全火力"]:
        try:
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            dfs = pd.read_html(StringIO(requests.get(url, headers=headers).text))
            for df in dfs:
                col = [c for c in df.columns if 'Ticker' in c or 'Symbol' in c]
                if col and len(df) > 90:
                    all_tickers.extend(df[col[0]].tolist())
                    break
        except: pass

    # 清理格式
    clean_list = list(set([str(t).replace('.', '-') for t in all_tickers if len(str(t)) < 6]))
    
    # 緊急備份清單 (萬一 Wikipedia 封鎖 IP)
    if not clean_list:
        return ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "PLTR", "TSM", "AVGO", "NFLX"]
        
    return clean_list

def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol); df_1h = stock.history(period="6mo", interval="1h")
        if len(df_1h) < 240: return None
        df_daily = df_1h.resample('D').agg({'Volume': 'sum', 'Close': 'last'}).dropna()
        df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
        
        # 濾網 1: 均線趨勢
        if check_daily_ma60_up and df_daily['MA60'].iloc[-1] <= df_daily['MA60'].iloc[-2]: return None
        if check_price_above_daily_ma60 and df_daily['Close'].iloc[-1] < df_daily['MA60'].iloc[-1]: return None
        
        # 濾網 2: 成交量
        if df_daily['Volume'].rolling(20).mean().iloc[-1] < vol_threshold: return None
        
        # 濾網 3: HV Rank
        log_ret = np.log(df_daily['Close'] / df_daily['Close'].shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        hv_rank = ((vol_30d.iloc[-1] - vol_30d.min()) / (vol_30d.max() - vol_30d.min())) * 100
        if hv_rank > hv_threshold: return None
        
        # 週波動計算
        week_vol_move = log_ret.tail(5).std() * np.sqrt(5) * 100 if len(log_ret) >= 5 else 0

        # 濾網 4: 4H 乖離與 U型
        df_4h = df_1h.resample('4h').agg({'Close': 'last'}).dropna()
        df_4h['MA60'] = df_4h['Close'].rolling(60).mean()
        dist_pct = ((df_4h['Close'].iloc[-1] - df_4h['MA60'].iloc[-1]) / df_4h['MA60'].iloc[-1]) * 100
        if abs(dist_pct) > dist_threshold: return None 
        
        u_score = -abs(dist_pct)
        if enable_u_logic:
            y = df_4h['MA60'].tail(u_sensitivity).values; x = np.arange(len(y)); coeffs = np.polyfit(x, y, 2)
            if coeffs[0] > 0 and (len(y)*0.3 <= -coeffs[1]/(2*coeffs[0]) <= len(y)*1.1) and (y[-1]-y[-2]) > 0 and coeffs[0] >= min_curvature:
                u_score = (coeffs[0] * 1000) - (abs(dist_pct) * 0.5)
            else: return None

        return {
            "代號": symbol, "HV Rank": round(hv_rank, 1), "週波動%": round(week_vol_move, 2),
            "現價": round(df_4h['Close'].iloc[-1], 2), "4H 60MA": round(df_4h['MA60'].iloc[-1], 2),
            "乖離率": f"{round(dist_pct, 2)}%", "產業": stock.info.get('industry', 'N/A'),
            "財報日": stock.calendar['Earnings Date'][0].strftime('%m-%d') if stock.calendar and 'Earnings Date' in stock.calendar else "未知",
            "題材搜尋": f"https://www.google.com/search?q={symbol}+題材+風險", "_sort_score": u_score
        }
    except: return None

# --- 6. 掃描與結果顯示 ---
if st.button("🚀 啟動 Turbo 掃描", type="primary"):
    st.session_state['scan_results'] = None
    min_volume_threshold = min_vol_m * 1000000 
    
    with st.status("🔍 掃描器診斷中...", expanded=True) as status:
        tickers = get_tickers_robust(market_choice)[:scan_limit]
        status.write(f"✅ 已成功抓取 {len(tickers)} 檔市場代號。")
        
        results = []; progress = st.progress(0); count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(get_ghost_metrics, t, min_volume_threshold): t for t in tickers}
            for future in as_completed(future_to_ticker):
                data = future.result(); count += 1; progress.progress(count / len(tickers))
                if data: results.append(data)
        
        st.session_state['scan_results'] = results
        if not results:
            status.update(label="⚠️ 掃描完成，但沒有符合條件的股票。", state="complete", expanded=True)
            st.warning("診斷提示：目前濾網可能太嚴格（例如 HV Rank < 30 或日均量 > 10M），建議試著調高 HV 門檻再掃一次。")
        else:
            status.update(label=f"🎯 掃描完成！共發現 {len(results)} 檔優質標的。", state="complete", expanded=False)

if 'scan_results' in st.session_state and st.session_state['scan_results']:
    df = pd.DataFrame(st.session_state['scan_results']).sort_values(by="HV Rank", ascending=True)
    st.dataframe(df, column_config={
        "代號": st.column_config.LinkColumn("代號", display_text="Yahoo", help="點擊開啟 Yahoo Finance"),
        "題材搜尋": st.column_config.LinkColumn("題材", display_text="🔍 查詢")
    }, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    selected = st.selectbox("🕯️ 選擇股票檢視 K 線:", df.apply(lambda x: f"{x['代號']} - {x['產業']}", axis=1).tolist())
    if selected:
        # 此處呼叫 plot_interactive_chart 繪圖邏輯 (保持原本優化好的三週期圖表即可)
        pass # 原繪圖函數代碼較長，請保持原本 goast.py 內的 plot_interactive_chart 邏輯
