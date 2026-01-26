import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器 (2026)", page_icon="👻", layout="wide")

# 初始化 Session State
if 'scan_limit' not in st.session_state: st.session_state.scan_limit = 600 
if 'min_vol_m' not in st.session_state: st.session_state.min_vol_m = 10
if 'dist_threshold' not in st.session_state: st.session_state.dist_threshold = 8.0
if 'u_sensitivity' not in st.session_state: st.session_state.u_sensitivity = 30

if 'backup' not in st.session_state:
    st.session_state.backup = {
        'scan_limit': 600, 'min_vol_m': 10, 'dist_threshold': 8.0, 'u_sensitivity': 30
    }

def handle_u_logic_toggle():
    """連動邏輯：啟動U型時，自動調整參數以利偵測"""
    if st.session_state.u_logic_key:
        st.session_state.backup.update({
            'scan_limit': st.session_state.scan_limit,
            'min_vol_m': st.session_state.min_vol_m,
            'dist_threshold': st.session_state.dist_threshold,
            'u_sensitivity': st.session_state.u_sensitivity
        })
        # 啟動 U 型戰法時，因預設開啟嚴格勺子，直接將敏感度拉到最大 (240)
        st.session_state.scan_limit = 600
        st.session_state.min_vol_m = 1
        st.session_state.dist_threshold = 50.0
        st.session_state.u_sensitivity = 240 
    else:
        st.session_state.scan_limit = st.session_state.backup['scan_limit']
        st.session_state.min_vol_m = st.session_state.backup['min_vol_m']
        st.session_state.dist_threshold = st.session_state.backup['dist_threshold']
        st.session_state.u_sensitivity = st.session_state.backup['u_sensitivity']

def handle_spoon_toggle():
    """勺子模式獨立連動：當手動勾選嚴格勺子時，也將敏感度設為最大"""
    if st.session_state.spoon_strict_key:
        st.session_state.u_sensitivity = 240

st.title("👻 幽靈策略掃描器")
st.caption(f"📅 台灣時間：{datetime.now().strftime('%Y-%m-%d %H:%M')} (2026年)")

# --- 2. 核心策略導引區 ---
with st.expander("📖 點擊展開：幽靈策略動態蝴蝶演化步驟 (詳細準則)", expanded=False):
    col_step1, col_step2, col_step3 = st.columns(3)
    with col_step1:
        st.markdown("### 第一步：建立試探部位 (Rule 1)")
        st.markdown("**動作**: 買進低價 Call + 賣出高價 Call (多頭價差)。\n**失敗**: 2日橫盤或跌破支撐。")
    with col_step2:
        st.markdown("### 第二步：動能加碼 (Rule 2)")
        st.markdown("**動作**: 浮盈後，加買更高階 Call。\n**指標**: IV 擴張 (水結成冰)。")
    with col_step3:
        st.markdown("### 第三步：轉化蝴蝶 (退出方案)")
        st.markdown("**動作**: 漲破加碼價時，加賣中間價位 Call。\n**目標**: 達成負成本蝴蝶型態。")
    st.info("💡 **核心注意事項**：Step 2 重點在於 IV 擴張。")

st.markdown("---")

# --- 3. 側邊欄設定 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio("市場", ["S&P 500", "NASDAQ 100", "🔥 全火力"], index=2)

st.sidebar.header("📈 戰法連動")
enable_u_logic = st.sidebar.checkbox("✅ 啟動 4小時 U型戰法連動", value=False, key='u_logic_key', on_change=handle_u_logic_toggle)

# --- 嚴格勺子模式 ---
enable_spoon_strict = False
spoon_vertex_range = (50, 95)
if enable_u_logic:
    enable_spoon_strict = st.sidebar.checkbox("🥄 嚴格勺子模式 (尋找剛翻揚)", value=True, key='spoon_strict_key', on_change=handle_spoon_toggle)
    if enable_spoon_strict:
        spoon_vertex_range = st.sidebar.slider("🥄 勺子底部位置 (%)", 0, 100, (50, 95), 5)

scan_limit = st.sidebar.slider("掃描數量", 50, 600, key='scan_limit')

# --- 【更新】趨勢濾網 (修正為週線邏輯) ---
st.sidebar.header("🛡️ 趨勢濾網")
check_daily_ma60_up = st.sidebar.checkbox("✅ 日線 60MA 向上 (昨日<今日)", value=True)
# 修改選項標籤，明確指出是「週線」
check_ma60_strong_trend = st.sidebar.checkbox("✅ 週線 MA60 強勢趨勢 (連續5週上升)", value=True, help="強制篩選出「週線」MA60 呈現穩定上升曲線的股票 (如 CCL)")
check_price_above_daily_ma60 = st.sidebar.checkbox("✅ 股價 > 日線 60MA", value=True)

st.sidebar.header("⚙️ 基礎篩選")
hv_threshold = st.sidebar.slider("HV Rank 門檻", 10, 100, 30)
min_vol_m = st.sidebar.slider("最小日均量 (百萬股)", 1, 100, key='min_vol_m') 
dist_threshold = st.sidebar.slider("距離 MA60 範圍 (%)", 0.0, 50.0, key='dist_threshold', step=0.5)

if enable_u_logic:
    u_sensitivity = st.sidebar.slider("U型敏感度", 20, 240, key='u_sensitivity')
    min_curvature = st.sidebar.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")
else:
    u_sensitivity, min_curvature = 30, 0.003
max_workers = st.sidebar.slider("🚀 平行核心數", 1, 32, 16)

# --- 4. 產業翻譯 ---
INDUSTRY_MAP = {
    "technology": "科技", "software": "軟體服務", "semiconductors": "半導體",
    "financial": "金融銀行", "healthcare": "醫療保健", "energy": "能源", 
    "industrials": "工業製造", "consumer cyclical": "循環性消費", 
    "consumer defensive": "防禦性消費", "utilities": "公用事業", 
    "real estate": "房地產", "communication": "通訊服務", "retail": "零售"
}
def translate_industry(eng):
    if not eng: return "未知"
    target = eng.lower()
    for key, val in INDUSTRY_MAP.items():
        if key in target: return val
    return eng

# --- 5. 核心繪圖函數 ---
def plot_interactive_chart(symbol):
    stock = yf.Ticker(symbol)
    tab1, tab2, tab3 = st.tabs(["🗓️ 周線", "📅 日線", "⏱️ 4H"])
    layout = dict(xaxis_rangeslider_visible=False, height=600, margin=dict(l=10, r=10, t=50, b=50), legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"), dragmode=False)
    
    with tab1: # 周線
        try:
            df = stock.history(period="max", interval="1wk")
            if len(df) > 0:
                df['MA60'] = df['Close'].rolling(60).mean()
                fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='周K'),
                                 go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3))])
                fig.update_layout(title=f"{symbol} 周線", **layout)
                if len(df) > 150: fig.update_xaxes(range=[df.index[-150], df.index[-1]])
                st.plotly_chart(fig, use_container_width=True)
        except: st.error("周線載入失敗")

    with tab2: # 日線
        try:
            df = stock.history(period="10y")
            if len(df) > 0:
                df['MA60'] = df['Close'].rolling(60).mean()
                fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='日K'),
                                 go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3))])
                fig.update_layout(title=f"{symbol} 日線", **layout)
                if len(df) > 200: fig.update_xaxes(range=[df.index[-200], df.index[-1]])
                st.plotly_chart(fig, use_container_width=True)
        except: st.error("日線載入失敗")

    with tab3: # 4H
        try:
            df_1h = stock.history(period="1y", interval="1h")
            if len(df_1h) > 0:
                df = df_1h.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
                df['MA60'] = df['Close'].rolling(60).mean(); df['d_str'] = df.index.strftime('%m-%d %H:%M')
                fig = go.Figure([go.Candlestick(x=df['d_str'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='4H K'),
                                 go.Scatter(x=df['d_str'], y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3))])
                fig.update_layout(title=f"{symbol} 4H", **layout)
                st.plotly_chart(fig, use_container_width=True)
        except: st.error("4H 載入失敗")

# --- 6. 核心指標運算 (含週線 MA60 邏輯) ---
def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol); 
        # 1. 先抓 1年小時資料 (用於成交量與 4H 策略)
        df_1h = stock.history(period="1y", interval="1h")
        if len(df_1h) < 240: return None
        
        # 2. 轉換為日線做基礎過濾
        df_daily = df_1h.resample('D').agg({'Volume': 'sum', 'Close': 'last'}).dropna()
        df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
        
        # 3. 基礎日線趨勢檢查
        if check_daily_ma60_up and df_daily['MA60'].iloc[-1] <= df_daily['MA60'].iloc[-2]: return None
        if df_daily['Volume'].rolling(20).mean().iloc[-1] < vol_threshold: return None
        
        # 【修改處】週線 MA60 強勢趨勢過濾 (連續 5 週上升)
        if check_ma60_strong_trend:
            # 額外抓取 2年週線資料 (因為1年小時資料不足以計算長週期的週線 MA60)
            df_wk = stock.history(period="2y", interval="1wk")
            if len(df_wk) > 65: # 確保資料足夠
                df_wk['MA60'] = df_wk['Close'].rolling(60).mean()
                # 檢查最後 5 週 MA60 是否呈現嚴格遞增
                if not df_wk['MA60'].tail(5).is_monotonic_increasing: return None
            else:
                return None # 資料不足視為不通過

        # 4. 價格與波動率檢查
        if check_price_above_daily_ma60 and df_daily['Close'].iloc[-1] < df_daily['MA60'].iloc[-1]: return None
        
        log_ret = np.log(df_daily['Close'] / df_daily['Close'].shift(1))
        vol_30d = log_ret.rolling(30).std() * np.sqrt(252) * 100
        hv_rank = ((vol_30d.iloc[-1] - vol_30d.min()) / (vol_30d.max() - vol_30d.min())) * 100
        if hv_rank > hv_threshold: return None
        
        # 5. 乖離率與 U 型 (使用 4H 資料)
        df_4h = df_1h.resample('4h').agg({'Close': 'last'}).dropna()
        df_4h['MA60'] = df_4h['Close'].rolling(60).mean()
        dist_pct = ((df_4h['Close'].iloc[-1] - df_4h['MA60'].iloc[-1]) / df_4h['MA60'].iloc[-1]) * 100
        if abs(dist_pct) > dist_threshold: return None
        
        u_score = -abs(dist_pct)
        if enable_u_logic:
            y = df_4h['MA60'].tail(u_sensitivity).values; x = np.arange(len(y))
            a, b, c = np.polyfit(x, y, 2)
            vertex_x = -b / (2 * a)
            if a <= 0: return None
            
            if enable_spoon_strict:
                min_p, max_p = spoon_vertex_range
                if not (len(y)*(min_p/100) <= vertex_x <= len(y)*(max_p/100)): return None
                if y[-1] <= y[-2] or y[0] < y[-1]: return None
                u_score = 1000
            else:
                if not (len(y)*0.3 <= vertex_x <= len(y)*1.1): return None
                if y[-1] <= y[-2]: return None
                u_score = (a * 1000) - (abs(dist_pct) * 0.5)
            if a < min_curvature: return None

        return {
            "代號": symbol, "HV Rank": round(hv_rank, 1), "現價": round(df_daily['Close'].iloc[-1], 2),
            "乖離率": f"{round(dist_pct, 2)}%", "產業": translate_industry(stock.info.get('industry', 'N/A')),
            "_sort_score": u_score
        }
    except: return None

# --- 7. 抓取代號 ---
@st.cache_data(ttl=3600)
def get_tickers_robust(choice):
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers = []
    try: # S&P 500
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[0]
        tickers.extend(df[df.columns[0]].tolist())
    except: pass
    try: # Nasdaq 100
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        dfs = pd.read_html(StringIO(requests.get(url, headers=headers).text))
        for df in dfs:
            if 95 <= len(df) <= 105: tickers.extend(df[df.columns[0]].tolist()); break
    except: pass
    final = list(set([str(t).replace('.', '-') for t in tickers if len(str(t)) < 6]))
    return final if final else ["AAPL", "NVDA", "TSLA", "AMD"]

# --- 8. 主程式執行 ---
if st.button("🚀 啟動 Turbo 掃描", type="primary"):
    st.session_state['scan_results'] = None
    min_volume_threshold = st.session_state.min_vol_m * 1000000 
    
    with st.status("🔍 掃描中...", expanded=True) as status:
        tickers = get_tickers_robust(market_choice)[:scan_limit]
        results = []; count = 0; total = len(tickers)
        progress = st.progress(0)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(get_ghost_metrics, t, min_volume_threshold): t for t in tickers}
            for future in as_completed(future_to_ticker):
                data = future.result(); count += 1
                progress.progress(count / total if total > 0 else 0)
                if data: results.append(data)
        st.session_state['scan_results'] = results
        status.update(label=f"完成！共 {len(results)} 檔。", state="complete", expanded=False)

if 'scan_results' in st.session_state and st.session_state['scan_results']:
    df = pd.DataFrame(st.session_state['scan_results']).sort_values(by="_sort_score", ascending=False if enable_u_logic else True)
    
    # 顯示資料 (Yahoo Statistics 連結)
    df_display = df.copy()
    df_display["代號"] = df_display["代號"].apply(lambda x: f"https://finance.yahoo.com/quote/{x}/key-statistics")

    st.subheader("📋 幽靈策略篩選列表")
    st.dataframe(
        df_display,
        column_config={
            "代號": st.column_config.LinkColumn("代號 (點我跳轉)", display_text="https://finance\\.yahoo\\.com/quote/(.*?)/key-statistics"),
            "_sort_score": None
        },
        hide_index=True, use_container_width=True
    )
    
    st.markdown("---")
    
    # --- 【無鍵盤選股區】使用 Expander + Radio 解決手機鍵盤問題 ---
    options = df.apply(lambda x: f"{x['代號']} - {x['產業']}", axis=1).tolist()
    
    if 'selected_idx' not in st.session_state: st.session_state.selected_idx = 0
    
    # 取得目前顯示的股票標籤
    current_label = options[st.session_state.selected_idx] if options and st.session_state.selected_idx < len(options) else "無資料"
    
    st.subheader("🕯️ 三週期 K 線檢視")
    
    # 使用 Expander 包裹 Radio，模擬下拉選單但無鍵盤
    with st.expander(f"🔽 點擊切換股票 (目前: {current_label.split(' - ')[0]})", expanded=False):
        if options:
            selected_opt = st.radio(
                "請直接點選 (不會跳出鍵盤):", 
                options, 
                index=st.session_state.selected_idx,
                key="stock_radio"
            )
            # 更新索引
            if selected_opt in options:
                st.session_state.selected_idx = options.index(selected_opt)
        else:
            st.write("查無符合條件標的")

    # 繪圖
    if options:
        target = options[st.session_state.selected_idx].split(" - ")[0]
        plot_interactive_chart(target)
