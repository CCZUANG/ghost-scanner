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
        # 【修改】啟動 U 型戰法時，因預設開啟嚴格勺子，直接將敏感度拉到最大 (240)
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
    """【新增】勺子模式獨立連動：當手動勾選嚴格勺子時，也將敏感度設為最大"""
    if st.session_state.spoon_strict_key:
        st.session_state.u_sensitivity = 240

st.title("👻 幽靈策略掃描器")
st.caption(f"📅 台灣時間：{datetime.now().strftime('%Y-%m-%d %H:%M')} (2026年)")

# --- 2. 核心策略導引區 (Step 1-3 詳細準則 - 排版優化版) ---
with st.expander("📖 點擊展開：幽靈策略動態蝴蝶演化步驟 (詳細準則)", expanded=False):
    col_step1, col_step2, col_step3 = st.columns(3)
    
    with col_step1:
        st.markdown("### 第一步：建立試探部位 (Rule 1)")
        st.markdown("""
        **🚀 啟動時機**
        放量突破關鍵壓力或回測支撐成功時。

        **動作**
        買進 **低價位 Call** + 賣出 **高一階 Call** (**多頭價差**)。

        **成功指標**
        股價站穩成本區，$\Delta$ (Delta) 隨價格上升而穩定增加。

        **❌ 失敗判定**
        2 交易日橫盤或跌破支撐 / 總損失超過 3 點。
        """)
        
    with col_step2:
        st.markdown("### 第二步：動能加碼 (Rule 2)")
        st.markdown("""
        **🚀 啟動時機**
        當價差已產生「浮盈」，且股價衝向賣出價位時。

        **動作**
        加買 **更高一階的 Call**。

        **成功指標**
        IV 顯著擴張（**水結成冰**），部位因波動迅速膨脹。

        **❌ 失敗判定**
        動能衰竭或 IV 下降（冰塊融化）。
        """)
        
    with col_step3:
        st.markdown("### 第三步：轉化蝴蝶 (退出方案)")
        st.markdown("""
        **🚀 啟動時機**
        股價強勢漲破加碼價，且市場出現過熱訊號時。

        **動作**
        **再加賣一張中間價位的 Call** (總計賣出兩張)。

        **成功指標**
        型態轉為 **蝴蝶型態 (+1/-2/+1)**，達成負成本。

        **❌ 失敗判定**
        爆量不漲或價格遠超最高階。
        """)

    st.info("💡 **核心注意事項**：Step 2 重點在於 IV 擴張。只有在部位已「證明你是對的」時才能執行 Rule 2 加碼。")

st.markdown("---")

# --- 3. 側邊欄 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio("市場", ["S&P 500", "NASDAQ 100", "🔥 全火力"], index=2)

st.sidebar.header("📈 戰法連動")
enable_u_logic = st.sidebar.checkbox("✅ 啟動 4小時 U型戰法連動", value=False, key='u_logic_key', on_change=handle_u_logic_toggle)

# --- 嚴格勺子模式與範圍設定 ---
enable_spoon_strict = False
spoon_vertex_range = (50, 95) # 預設值

if enable_u_logic:
    # 【修改】加入 key='spoon_strict_key' 與 on_change=handle_spoon_toggle
    enable_spoon_strict = st.sidebar.checkbox(
        "🥄 嚴格勺子模式 (尋找剛翻揚)", 
        value=True, 
        key='spoon_strict_key',
        on_change=handle_spoon_toggle,
        help="強制要求 MA60 的最低點發生在近期，排除已經漲很多的股票。"
    )
    
    if enable_spoon_strict:
        spoon_vertex_range = st.sidebar.slider(
            "🥄 勺子底部發生位置 (%)",
            min_value=0, 
            max_value=100, 
            value=(50, 95), 
            step=5,
            help="設定拋物線最低點(Vertex)必須落在回測期間的哪個百分比區段。"
        )

scan_limit = st.sidebar.slider("掃描數量", 50, 600, key='scan_limit')

st.sidebar.header("🛡️ 趨勢濾網")
check_daily_ma60_up = st.sidebar.checkbox("✅ 日線 60MA 向上", value=True)
check_price_above_daily_ma60 = st.sidebar.checkbox("✅ 股價 > 日線 60MA", value=True)

st.sidebar.header("⚙️ 基礎篩選")
hv_threshold = st.sidebar.slider("HV Rank 門檻", 10, 100, 30)
min_vol_m = st.sidebar.slider("最小日均量 (百萬股)", 1, 100, key='min_vol_m') 
dist_threshold = st.sidebar.slider("距離 MA60 範圍 (%)", 0.0, 50.0, key='dist_threshold', step=0.5)

if enable_u_logic:
    # 【修改】最大值調整為 240
    u_sensitivity = st.sidebar.slider("U型敏感度 (Lookback)", 20, 240, key='u_sensitivity')
    min_curvature = st.sidebar.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")
else:
    u_sensitivity, min_curvature = 30, 0.003
max_workers = st.sidebar.slider("🚀 平行核心數", 1, 32, 16)

# --- 4. 產業翻譯 ---
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

# --- 5. 核心繪圖函數 ---
def plot_interactive_chart(symbol):
    stock = yf.Ticker(symbol)
    tab1, tab2, tab3 = st.tabs(["🗓️ 周線", "📅 日線", "⏱️ 4H"])
    layout = dict(xaxis_rangeslider_visible=False, height=600, margin=dict(l=10, r=10, t=50, b=50), legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"), dragmode=False)
    config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}

    with tab1: # 周線 (max)
        try:
            df = stock.history(period="max", interval="1wk")
            if len(df) > 0:
                df['MA60'] = df['Close'].rolling(60).mean()
                fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='周K'),
                                 go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3))])
                fig.update_layout(title=dict(text=f"{symbol} 周線 (全歷史)", x=0.02), **layout)
                if len(df) > 150: fig.update_xaxes(range=[df.index[-150], df.index[-1]])
                st.plotly_chart(fig, use_container_width=True, config=config)
            else: st.warning("周線無數據")
        except Exception as e: st.error(f"周線圖錯誤: {e}")

    with tab2: # 日線 (10y)
        try:
            df = stock.history(period="10y")
            if len(df) > 0:
                df['MA60'] = df['Close'].rolling(60).mean()
                fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='日K'),
                                 go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3))])
                fig.update_layout(title=dict(text=f"{symbol} 日線 (10年)", x=0.02), **layout); fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                if len(df) > 200: fig.update_xaxes(range=[df.index[-200], df.index[-1]])
                st.plotly_chart(fig, use_container_width=True, config=config)
            else: st.warning("日線無數據")
        except Exception as e: st.error(f"日線圖錯誤: {e}")

    with tab3: # 4H (1y)
        try:
            df_1h = stock.history(period="1y", interval="1h")
            if len(df_1h) > 0:
                df = df_1h.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
                df['MA60'] = df['Close'].rolling(60).mean(); df['date_str'] = df.index.strftime('%m-%d %H:%M')
                fig = go.Figure([go.Candlestick(x=df['date_str'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='4H K'),
                                 go.Scatter(x=df['date_str'], y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3), connectgaps=True)])
                fig.update_layout(title=dict(text=f"{symbol} 4小時圖 (1年)", x=0.02), **layout); fig.update_xaxes(type='category', range=[max(0, len(df)-160), len(df)])
                st.plotly_chart(fig, use_container_width=True, config=config)
            else: st.warning("4H 無數據")
        except Exception as e: st.error(f"4H 圖錯誤: {e}")

# --- 6. 核心指標運算 (含勺子邏輯) ---
def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol); df_1h = stock.history(period="1y", interval="1h")
        if len(df_1h) < 240: return None
        df_daily = df_1h.resample('D').agg({'Volume': 'sum', 'Close': 'last'}).dropna()
        df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
        
        if check_daily_ma60_up and df_daily['MA60'].iloc[-1] <= df_daily['MA60'].iloc[-2]: return None
        if check_price_above_daily_ma60 and df_daily['Close'].iloc[-1] < df_daily['MA60'].iloc[-1]: return None
        if df_daily['Volume'].rolling(20).mean().iloc[-1] < vol_threshold: return None
        
        log_ret = np.log(df_daily['Close'] / df_daily['Close'].shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        hv_rank = ((vol_30d.iloc[-1] - vol_30d.min()) / (vol_30d.max() - vol_30d.min())) * 100
        if hv_rank > hv_threshold: return None
        
        week_vol_move = log_ret.tail(5).std() * np.sqrt(5) * 100 if len(log_ret) >= 5 else 0
        cur_price = df_daily['Close'].iloc[-1]
        move_dollar = cur_price * (week_vol_move / 100)

        df_4h = df_1h.resample('4h').agg({'Close': 'last'}).dropna()
        df_4h['MA60'] = df_4h['Close'].rolling(60).mean()
        dist_pct = ((df_4h['Close'].iloc[-1] - df_4h['MA60'].iloc[-1]) / df_4h['MA60'].iloc[-1]) * 100
        if abs(dist_pct) > dist_threshold: return None 
        
        u_score = -abs(dist_pct)
        if enable_u_logic:
            y = df_4h['MA60'].tail(u_sensitivity).values
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 2)
            a, b, c = coeffs
            vertex_x = -b / (2 * a)
            
            if a <= 0: return None # 開口必須向上
            
            # --- 嚴格勺子邏輯 (動態參數化) ---
            if enable_spoon_strict:
                # 將百分比 (0-100) 轉為小數 (0.0-1.0)
                min_pos_pct = spoon_vertex_range[0] / 100.0
                max_pos_pct = spoon_vertex_range[1] / 100.0
                
                if not (len(y) * min_pos_pct <= vertex_x <= len(y) * max_pos_pct): return None
                
                if y[-1] <= y[-2]: return None
                if y[0] < y[-1]: return None 
                u_score = 1000
            else:
                if not (len(y) * 0.3 <= vertex_x <= len(y) * 1.1): return None
                if y[-1] <= y[-2]: return None
                u_score = (a * 1000) - (abs(dist_pct) * 0.5)
            
            if a < min_curvature: return None
        
        earnings_date = "未知"
        cal = stock.calendar
        if cal is not None and 'Earnings Date' in cal:
            earnings_date = cal['Earnings Date'][0].strftime('%m-%d')
            
        return {
            "代號": symbol, "HV Rank": round(hv_rank, 1), "週波動%": round(week_vol_move, 2),
            "預期變動$": f"±{round(move_dollar, 2)}", "現價": round(cur_price, 2),
            "4H 60MA": round(df_4h['MA60'].iloc[-1], 2), "乖離率": f"{round(dist_pct, 2)}%",
            "產業": translate_industry(stock.info.get('industry', 'N/A')),
            "下次財報": earnings_date, "題材搜尋": f"https://www.google.com/search?q={symbol}+題材+風險", "_sort_score": u_score
        }
    except: return None

# --- 7. 市場代號抓取 ---
@st.cache_data(ttl=3600)
def get_tickers_robust(choice):
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers = []
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        res = requests.get(url, headers=headers); df = pd.read_html(StringIO(res.text))[0]
        col = [c for c in df.columns if 'Symbol' in c or 'Ticker' in c][0]; tickers.extend(df[col].tolist())
    except: pass
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        res = requests.get(url, headers=headers); dfs = pd.read_html(StringIO(res.text))
        for df in dfs:
            col = [c for c in df.columns if 'Ticker' in c or 'Symbol' in c]
            if col and 95 <= len(df) <= 105:
                tickers.extend(df[col[0]].tolist()); break
    except: pass
    final = list(set([str(t).replace('.', '-') for t in tickers if len(str(t)) < 6]))
    return final if final else ["AAPL", "NVDA", "TSLA", "PLTR", "AMD"]

# --- 8. 主程式執行 ---
if st.button("🚀 啟動 Turbo 掃描", type="primary"):
    st.session_state['scan_results'] = None
    min_volume_threshold = st.session_state.min_vol_m * 1000000 
    
    with st.status("🔍 市場掃描中...", expanded=True) as status:
        tickers = get_tickers_robust(market_choice)[:scan_limit]
        total_tickers = len(tickers)
        status.write(f"✅ 已獲得 {total_tickers} 檔代號，開始技術面過濾...")
        results = []; progress = st.progress(0); count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(get_ghost_metrics, t, min_volume_threshold): t for t in tickers}
            for future in as_completed(future_to_ticker):
                data = future.result(); count += 1
                progress.progress(count / total_tickers if total_tickers > 0 else 0)
                if data: results.append(data)
        st.session_state['scan_results'] = results
        status.update(label=f"掃描完成！共發現 {len(results)} 檔標的。", state="complete", expanded=False)

if 'scan_results' in st.session_state and st.session_state['scan_results']:
    # 原始資料 (用於邏輯運算與圖表)
    df = pd.DataFrame(st.session_state['scan_results']).sort_values(by="_sort_score", ascending=False if enable_u_logic else True)
    
    # 【修改處 1】建立一個專門用於顯示的 DataFrame，將代號轉換為 Yahoo Financials 連結
    df_display = df.copy()
    df_display["代號"] = df_display["代號"].apply(lambda x: f"https://finance.yahoo.com/quote/{x}/financials")

    st.subheader("📋 幽靈策略篩選列表")
    
    # 【修改處 2】使用 LinkColumn 配合 Regex，讓表格顯示代號但連結到財報
    st.dataframe(df_display, column_config={
        "代號": st.column_config.LinkColumn(
            "代號", 
            display_text="https://finance\\.yahoo\\.com/quote/(.*?)/financials"  # 正則表達式：只顯示代號，隱藏網址
        ),
        "題材搜尋": st.column_config.LinkColumn("題材與風險", display_text="🔍 查詢"),
        "_sort_score": None
    }, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.info("💡 手機操作提示：圖表預設為鎖定狀態以利網頁捲動。如需平移或縮放 K 線，請點擊圖表右上角工具列的「十字箭頭 (Pan)」圖示解鎖。")
    st.subheader("🕯️ 三週期 K 線檢視")
    
    # 【修改處 3】下拉選單使用原始 df，確保抓取的是純代號 (如 NVDA) 而不是網址，避免繪圖錯誤
    selected = st.selectbox("選擇標的:", df.apply(lambda x: f"{x['代號']} - {x['產業']}", axis=1).tolist())
    if selected: plot_interactive_chart(selected.split(" - ")[0])
