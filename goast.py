import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# --- 1. 頁面與連動邏輯設定 ---
st.set_page_config(page_title="幽靈策略掃描器", page_icon="👻", layout="wide")

# 初始化 Session State (記憶還原系統)
if 'scan_limit' not in st.session_state: st.session_state.scan_limit = 600 
if 'min_vol_m' not in st.session_state: st.session_state.min_vol_m = 10
if 'dist_threshold' not in st.session_state: st.session_state.dist_threshold = 8.0
if 'u_sensitivity' not in st.session_state: st.session_state.u_sensitivity = 30

if 'backup' not in st.session_state:
    st.session_state.backup = {
        'scan_limit': 600, 'min_vol_m': 10, 'dist_threshold': 8.0, 'u_sensitivity': 30
    }

def handle_u_logic_toggle():
    """連動邏輯：啟動時備份設定，關閉時秒速還原"""
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

# --- 2. 核心策略導引區 (Step 1-3 完整說明) ---
st.write("**策略目標**：鎖定 **日線多頭 + 4H U型**，尋找「結冰區」起漲點。")

with st.expander("📖 幽靈策略：動態蝴蝶演化步驟 (詳細準則)", expanded=True):
    col_step1, col_step2, col_step3 = st.columns(3)
    with col_step1:
        st.markdown("### 第一步：建立試探 (Rule 1)")
        st.markdown("""
        **🚀 啟動時機**：放量突破關鍵壓力或回測支撐成功時。  
        **動作**：買進 低價位 Call + 賣出 高一階 Call (**多頭價差**)。  
        **成功指標**：股價站穩成本區，Delta 穩定增加。  
        **❌ 失敗判定**：2 交易日橫盤或跌破支撐。
        """)
    with col_step2:
        st.markdown("### 第二步：動能加碼 (Rule 2)")
        st.markdown("""
        **🚀 啟動時機**：價差已產生浮盈，且股價衝向賣出價位時。  
        **動作**：加買 **更高一階的 Call**。  
        **成功指標**：IV 顯著擴張 (**水結成冰**)，部位價值膨脹。  
        **❌ 失敗判定**：股價轉頭跌破成本區或 IV 下降。
        """)
    with col_step3:
        st.markdown("### 第三步：轉化蝴蝶 (退出方案)")
        st.markdown("""
        **🚀 啟動時機**：股價強勢漲破加碼價，且出現過熱訊號時。  
        **動作**：**再加賣一張中間價位 Call**。  
        **成功指標**：轉為蝴蝶型態達成 **負成本(穩賺)**。  
        """)
    st.info("💡 **核心注意事項**：只有在初始價差已經「證明你是對的」時才能執行 Rule 2 加碼。")

st.markdown("---")

# --- 3. 側邊欄：參數設定區 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio("市場", ["S&P 500", "NASDAQ 100", "🔥 全火力"], index=2)

st.sidebar.header("📈 戰法切換")
enable_u_logic = st.sidebar.checkbox("✅ 啟動 4小時 U型戰法連動", value=False, key='u_logic_key', on_change=handle_u_logic_toggle)
scan_limit = st.sidebar.slider("掃描數量", 50, 600, key='scan_limit')

st.sidebar.header("🛡️ 趨勢濾網")
check_daily_ma60_up = st.sidebar.checkbox("✅ 日線 60MA 向上", value=True)
check_price_above_daily_ma60 = st.sidebar.checkbox("✅ 股價 > 日線 60MA", value=True)

st.sidebar.header("⚙️ 基礎篩選")
hv_threshold = st.sidebar.slider("HV Rank 門檻", 10, 100, 30)
# 【修正】明確上限為 100 (一億股)
min_vol_m = st.sidebar.slider("最小日均量 (百萬股)", 1, 100, key='min_vol_m') 
min_volume_threshold = st.session_state.min_vol_m * 1000000

dist_threshold = st.sidebar.slider("距離 4H 60MA 範圍 (%)", 0.0, 50.0, key='dist_threshold', step=0.5)

if enable_u_logic:
    u_sensitivity = st.sidebar.slider("U型敏感度 (Lookback)", 20, 60, key='u_sensitivity')
    min_curvature = st.sidebar.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")
else:
    u_sensitivity, min_curvature = 30, 0.003
max_workers = st.sidebar.slider("🚀 運算核心數", 1, 32, 16)

# --- 4. 輔助函數：產業翻譯與繪圖 ---
INDUSTRY_MAP = {
    "technology": "科技", "software": "軟體服務", "semiconductors": "半導體",
    "financial": "金融銀行", "healthcare": "醫療保健", "biotechnology": "生物科技",
    "energy": "能源", "industrials": "工業製造", "consumer cyclical": "循環性消費",
    "consumer defensive": "防禦性消費", "utilities": "公用事業", "real estate": "房地產",
    "communication services": "通訊服務", "basic materials": "基礎原物料",
    "entertainment": "影視娛樂", "internet content": "網路內容", "auto": "汽車產業",
    "retail": "零售通路", "aerospace": "航太軍工", "banks": "銀行業"
}

def translate_industry(eng):
    if not eng or eng == "N/A": return "未知"
    target = eng.lower()
    for key, val in INDUSTRY_MAP.items():
        if key in target: return val
    return eng

def plot_interactive_chart(symbol):
    stock = yf.Ticker(symbol)
    tab1, tab2, tab3 = st.tabs(["🗓️ 周線", "📅 日線", "⏱️ 4H"])
    layout = dict(xaxis_rangeslider_visible=False, height=600, margin=dict(l=10, r=10, t=50, b=50), legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"), dragmode='pan')
    config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}

    with tab1: # 周線
        try:
            df = stock.history(period="5y", interval="1wk")
            df['MA60'] = df['Close'].rolling(60).mean()
            fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='周K'),
                             go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3))])
            fig.update_layout(title=dict(text=f"{symbol} 周線", x=0.02), **layout); fig.update_xaxes(range=[df.index[-100], df.index[-1]])
            st.plotly_chart(fig, use_container_width=True, config=config)
        except: pass
    with tab2: # 日線
        try:
            df = stock.history(period="2y")
            df['MA60'] = df['Close'].rolling(60).mean()
            fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='日K'),
                             go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3))])
            fig.update_layout(title=dict(text=f"{symbol} 日線", x=0.02), **layout); fig.update_xaxes(range=[df.index[-150], df.index[-1]], rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, use_container_width=True, config=config)
        except: pass
    with tab3: # 4H 無縫
        try:
            df_1h = stock.history(period="6mo", interval="1h")
            df = df_1h.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
            df['MA60'] = df['Close'].rolling(60).mean(); df['date_str'] = df.index.strftime('%m-%d %H:%M')
            fig = go.Figure([go.Candlestick(x=df['date_str'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='4H K'),
                             go.Scatter(x=df['date_str'], y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3), connectgaps=True)])
            fig.update_layout(title=dict(text=f"{symbol} 4小時圖", x=0.02), **layout); fig.update_xaxes(type='category', range=[max(0, len(df)-160), len(df)])
            st.plotly_chart(fig, use_container_width=True, config=config)
        except: pass

def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol); df_1h = stock.history(period="6mo", interval="1h")
        if len(df_1h) < 240: return None
        df_daily = df_1h.resample('D').agg({'Volume': 'sum', 'Close': 'last'}).dropna()
        df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
        if len(df_daily) < 60: return None
        if check_daily_ma60_up and df_daily['MA60'].iloc[-1] <= df_daily['MA60'].iloc[-2]: return None
        if check_price_above_daily_ma60 and df_daily['Close'].iloc[-1] < df_daily['MA60'].iloc[-1]: return None
        if df_daily['Volume'].rolling(20).mean().iloc[-1] < vol_threshold: return None
        
        log_ret = np.log(df_daily['Close'] / df_daily['Close'].shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        hv_rank = ((vol_30d.iloc[-1] - vol_30d.min()) / (vol_30d.max() - vol_30d.min())) * 100
        if hv_rank > hv_threshold: return None
        week_vol_move = log_ret.tail(5).std() * np.sqrt(5) * 100 if len(log_ret) >= 5 else 0

        df_4h = df_1h.resample('4h').agg({'Close': 'last'}).dropna()
        df_4h['MA60'] = df_4h['Close'].rolling(60).mean()
        dist_pct = ((df_4h['Close'].iloc[-1] - df_4h['MA60'].iloc[-1]) / df_4h['MA60'].iloc[-1]) * 100
        if abs(dist_pct) > dist_threshold: return None 
        u_score = -abs(dist_pct)
        if enable_u_logic:
            y = df_4h['MA60'].tail(u_sensitivity).values; coeffs = np.polyfit(np.arange(len(y)), y, 2)
            if coeffs[0] > 0 and (len(y)*0.3 <= -coeffs[1]/(2*coeffs[0]) <= len(y)*1.1) and (y[-1]-y[-2]) > 0 and coeffs[0] >= min_curvature:
                u_score = (coeffs[0] * 1000) - (abs(dist_pct) * 0.5)
            else: return None
            
        cal = stock.calendar; earnings = cal['Earnings Date'][0].strftime('%m-%d') if cal and 'Earnings Date' in cal else "未知"
        return {
            "代號": symbol, "HV Rank": round(hv_rank, 1), "週波動%": round(week_vol_move, 2),
            "現價": round(df_4h['Close'].iloc[-1], 2), "4H 60MA": round(df_4h['MA60'].iloc[-1], 2),
            "乖離率": f"{round(dist_pct, 2)}%", "產業": translate_industry(stock.info.get('industry', 'N/A')),
            "財報日": earnings, "題材搜尋": f"https://www.google.com/search?q={symbol}+美股+題材+風險", "_sort_score": u_score
        }
    except: return None

# --- 5. 掃描與執行 ---
@st.cache_data(ttl=3600)
def get_tickers_robust(choice):
    headers = {"User-Agent": "Mozilla/5.0"}
    all_tickers = []
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies" if "S&P" in choice or "全火力" in choice else "https://en.wikipedia.org/wiki/Nasdaq-100"
        df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[0 if "S&P" in choice else 2]
        col = [c for c in df.columns if 'Symbol' in c or 'Ticker' in c][0]
        all_tickers.extend(df[col].tolist())
    except: return ["AAPL", "NVDA", "TSM", "PLTR", "AMD"]
    return list(set([str(t).replace('.', '-') for t in all_tickers if len(str(t)) < 6]))

if st.button("🚀 啟動 Turbo 掃描", type="primary"):
    st.session_state['scan_results'] = None
    with st.status("🔍 掃描與診斷中...", expanded=True) as status:
        tickers = get_tickers_robust(market_choice)[:scan_limit]
        status.write(f"✅ 已抓取 {len(tickers)} 檔市場代號。")
        results = []; progress = st.progress(0); count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(get_ghost_metrics, t, min_volume_threshold): t for t in tickers}
            for future in as_completed(future_to_ticker):
                data = future.result(); count += 1; progress.progress(count / len(tickers))
                if data: results.append(data)
        st.session_state['scan_results'] = results
        status.update(label=f"完成！共發現 {len(results)} 檔。", state="complete", expanded=False)

if 'scan_results' in st.session_state and st.session_state['scan_results']:
    df = pd.DataFrame(st.session_state['scan_results']).sort_values(by="_sort_score", ascending=False)
    st.subheader("📋 幽靈策略篩選列表")
    st.dataframe(df, column_config={
        "代號": st.column_config.LinkColumn("代號", display_text="https://finance\\.yahoo\\.com/quote/(.*)"),
        "題材搜尋": st.column_config.LinkColumn("題材與風險", display_text="🔍 查詢"),
        "_sort_score": None
    }, hide_index=True, use_container_width=True)
    st.markdown("---")
    select_list = df.apply(lambda x: f"{x['代號']} - {x['產業']}", axis=1).tolist()
    selected = st.selectbox("🕯️ 選擇股票檢視 K 線:", select_list)
    if selected: plot_interactive_chart(selected.split(" - ")[0])
