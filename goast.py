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

st.title("👻 幽靈策略掃描器")

# --- 2. 核心策略導引區 ---
st.write("**策略目標**：鎖定 **日線多頭 + 4H U型**，尋找「結冰區」起漲點，並透過動態期權組合鎖定利潤。")

with st.expander("📖 幽靈策略：動態蝴蝶演化步驟", expanded=True):
    col_step1, col_step2, col_step3 = st.columns(3)
    
    with col_step1:
        st.subheader("第一步：建立試探部位")
        st.markdown("""
        **🚀 啟動時機**：正股放量突破關鍵壓力或回測支撐成功時。  
        **動作**：買進 **低價位 Call** + 賣出 **高一階 Call** (多頭價差)。  
        **成功指標**：股價站穩成本區，Delta 隨價格上升穩定增加。  
        **❌ 失敗判定**：
        - **時間**：進場後 2 個交易日股價橫盤。
        - **空間**：跌破支撐位或總損失超過 3 點。
        """)
        
    with col_step2:
        st.subheader("第二步：動能加碼")
        st.markdown("""
        **🚀 啟動時機**：價差已產生「浮盈」，且股價衝向賣出價時。  
        **動作**：加買 **更高一階的 Call**。  
        **成功指標**：IV 顯著擴張（水結成冰），部位體積因波動迅速膨脹。  
        **❌ 失敗判定**：
        - **動能**：股價觸及賣出價後轉頭跌破成本區。
        - **波動**：IV 下降（冰塊融化），加碼 Call 價值停滯。
        """)
        
    with col_step3:
        st.subheader("第三步：轉化蝴蝶")
        st.markdown("""
        **🚀 啟動時機**：股價強勢漲破加碼價，且出現過熱訊號時。  
        **動作**：**再加賣一張中間價位的 Call** (總計賣出兩張)。  
        **成功指標**：轉為 **蝴蝶型態 (+1/-2/+1)**，達成負成本（穩賺）。  
        **❌ 失敗判定**：
        - **爆量**：異常天量且價格停滯，三天內分批清空。
        - **結算**：價格遠超最高階且未見拉回，應獲利了結。
        """)

    st.info("💡 **核心注意事項**：Step 2 的靈魂在於 **IV 擴張**（水結成冰）。只有在價差部位已「證明你是對的」時才能執行 Rule 2 加碼。")

st.markdown("---")

# --- 3. 側邊欄：參數設定區 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio(
    "選擇掃描市場", 
    ["S&P 500 (大型股)", "NASDAQ 100 (科技股)", "🔥 全火力 (兩者全掃)"],
    index=2
)
scan_limit = st.sidebar.slider("掃描數量 (前 N 大)", 50, 600, 200)

st.sidebar.header("🛡️ 日線趨勢濾網")
check_daily_ma60_up = st.sidebar.checkbox("✅ 必須：日線 60MA 向上", value=True)
check_price_above_daily_ma60 = st.sidebar.checkbox("✅ 必須：股價 > 日線 60MA", value=True)

st.sidebar.header("⚙️ 基礎篩選")
hv_threshold = st.sidebar.slider("HV Rank 門檻 (越低越好)", 10, 100, 30)
min_vol_m = st.sidebar.slider("最小日均量 (百萬股)", 1, 100, 10) 
min_volume_threshold = min_vol_m * 1000000

st.sidebar.header("📈 4小時 U型戰法")
enable_u_logic = st.sidebar.checkbox("✅ 啟用「U型數學擬合」", value=True)
dist_threshold = st.sidebar.slider("距離 4H 60MA 範圍 (%)", 0.0, 50.0, 8.0, step=0.5)

if enable_u_logic:
    u_sensitivity = st.sidebar.slider("U型敏感度 (Lookback)", 20, 60, 30)
    min_curvature = st.sidebar.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")
else:
    u_sensitivity = 30
    min_curvature = 0.003

st.sidebar.markdown("---")
max_workers = st.sidebar.slider("🚀 平行運算核心數", 1, 32, 16)

# --- 4. 輔助與核心函數 ---

INDUSTRY_MAP = {
    "technology": "科技業", "software": "軟體", "semiconductors": "半導體",
    "financial": "金融", "banks": "銀行", "credit": "信貸",
    "healthcare": "醫療保健", "biotechnology": "生物科技",
    "consumer cyclical": "非必需消費", "auto": "汽車",
    "energy": "能源", "oil": "石油", "industrials": "工業",
    "aerospace": "航太軍工", "communication": "通訊", "internet": "網路",
    "utilities": "公用事業", "real estate": "房地產", "reit": "房地產信託",
    "basic materials": "原物料", "entertainment": "娛樂", "retail": "零售"
}

def translate_industry(eng_industry):
    if not eng_industry or eng_industry == "N/A": return "未知"
    target = str(eng_industry).lower().strip()
    if target in INDUSTRY_MAP: return INDUSTRY_MAP[target]
    for key, value in INDUSTRY_MAP.items():
        if key in target: return value
    return target.title()

def plot_interactive_chart(symbol):
    stock = yf.Ticker(symbol)
    tab1, tab2, tab3 = st.tabs(["🗓️ 周線 (Long)", "📅 日線 (Mid)", "⏱️ 4H (Short)"])
    
    layout_common = dict(
        xaxis_rangeslider_visible=False,
        height=600,  
        margin=dict(l=10, r=10, t=50, b=50), 
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        dragmode='pan', 
    )

    def get_title_config(text):
        return dict(text=text, x=0.02, xanchor='left', font=dict(size=16))

    config_common = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}

    # --- Tab 1: 周線 ---
    with tab1:
        try:
            df_w = stock.history(period="5y", interval="1wk")
            if len(df_w) > 60:
                df_w['MA20'] = df_w['Close'].rolling(window=20).mean()
                df_w['MA60'] = df_w['Close'].rolling(window=60).mean()
                fig_w = go.Figure()
                fig_w.add_trace(go.Candlestick(x=df_w.index, open=df_w['Open'], high=df_w['High'], low=df_w['Low'], close=df_w['Close'], name='周K'))
                fig_w.add_trace(go.Scatter(x=df_w.index, y=df_w['MA20'], mode='lines', name='MA20', line=dict(color='royalblue', width=1), connectgaps=True))
                fig_w.add_trace(go.Scatter(x=df_w.index, y=df_w['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3), connectgaps=True))
                fig_w.update_layout(title=get_title_config(f"{symbol} 周線"), yaxis_title="股價", **layout_common)
                if len(df_w) > 100:
                    fig_w.update_xaxes(range=[df_w.index[-100], df_w.index[-1] + pd.Timedelta(weeks=1)])
                st.plotly_chart(fig_w, use_container_width=True, config=config_common)
        except: pass

    # --- Tab 2: 日線 ---
    with tab2:
        try:
            df_d = stock.history(period="2y")
            if len(df_d) > 60:
                df_d['MA20'] = df_d['Close'].rolling(window=20).mean()
                df_d['MA60'] = df_d['Close'].rolling(window=60).mean()
                fig_d = go.Figure()
                fig_d.add_trace(go.Candlestick(x=df_d.index, open=df_d['Open'], high=df_d['High'], low=df_d['Low'], close=df_d['Close'], name='日K'))
                fig_d.add_trace(go.Scatter(x=df_d.index, y=df_d['MA20'], mode='lines', name='MA20', line=dict(color='royalblue', width=1), connectgaps=True))
                fig_d.add_trace(go.Scatter(x=df_d.index, y=df_d['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3), connectgaps=True))
                fig_d.update_layout(title=get_title_config(f"{symbol} 日線"), yaxis_title="股價", **layout_common)
                if len(df_d) > 150:
                    fig_d.update_xaxes(range=[df_d.index[-150], df_d.index[-1] + pd.Timedelta(days=2)], rangebreaks=[dict(bounds=["sat", "mon"])])
                st.plotly_chart(fig_d, use_container_width=True, config=config_common)
        except: pass

    # --- Tab 3: 4小時 (Category Axis) ---
    with tab3:
        try:
            df_1h = stock.history(period="6mo", interval="1h")
            df_4h = df_1h.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
            df_4h['MA20'] = df_4h['Close'].rolling(window=20).mean()
            df_4h['MA60'] = df_4h['Close'].rolling(window=60).mean()
            df_4h['date_str'] = df_4h.index.strftime('%m-%d %H:%M')
            fig_4h = go.Figure()
            fig_4h.add_trace(go.Candlestick(x=df_4h['date_str'], open=df_4h['Open'], high=df_4h['High'], low=df_4h['Low'], close=df_4h['Close'], name='4H K'))
            fig_4h.add_trace(go.Scatter(x=df_4h['date_str'], y=df_4h['MA20'], mode='lines', name='MA20', line=dict(color='royalblue', width=1), connectgaps=True))
            fig_4h.add_trace(go.Scatter(x=df_4h['date_str'], y=df_4h['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3), connectgaps=True))
            fig_4h.update_layout(title=get_title_config(f"{symbol} 4小時圖"), yaxis_title="股價", **layout_common)
            total_bars = len(df_4h)
            fig_4h.update_xaxes(type='category', range=[max(0, total_bars - 160), total_bars])
            st.plotly_chart(fig_4h, use_container_width=True, config=config_common)
        except: pass

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(StringIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text))[0]
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except: return []

@st.cache_data(ttl=3600)
def get_nasdaq100_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        dfs = pd.read_html(StringIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text))
        for df in dfs:
            if 'Ticker' in df.columns: return [t.replace('.', '-') for t in df['Ticker'].tolist()]
            elif 'Symbol' in df.columns: return [t.replace('.', '-') for t in df['Symbol'].tolist()]
        return []
    except: return []

def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol)
        df_1h = stock.history(period="6mo", interval="1h")
        if len(df_1h) < 240: return None
        df_daily = df_1h.resample('D').agg({'Volume': 'sum', 'Close': 'last'}).dropna()
        df_daily['MA60'] = df_daily['Close'].rolling(window=60).mean()
        if len(df_daily) < 60: return None
        if check_daily_ma60_up and df_daily['MA60'].iloc[-1] <= df_daily['MA60'].iloc[-2]: return None
        if check_price_above_daily_ma60 and df_daily['Close'].iloc[-1] < df_daily['MA60'].iloc[-1]: return None
        if df_daily['Volume'].rolling(window=20).mean().iloc[-1] < vol_threshold: return None
        log_ret = np.log(df_daily['Close'] / df_daily['Close'].shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        vol_5d = log_ret.tail(5).std() * np.sqrt(252) * 100 if len(log_ret) >= 5 else 0
        hv_rank = ((vol_30d.iloc[-1] - vol_30d.min()) / (vol_30d.max() - vol_30d.min())) * 100
        if hv_rank > hv_threshold: return None
        df_4h = df_1h.resample('4h').agg({'Close': 'last'}).dropna()
        df_4h['MA60'] = df_4h['Close'].rolling(window=60).mean()
        ma_segment = df_4h['MA60'].iloc[-u_sensitivity:]
        dist_pct = ((df_4h['Close'].iloc[-1] - ma_segment.iloc[-1]) / ma_segment.iloc[-1]) * 100
        if abs(dist_pct) > dist_threshold: return None 
        u_score = -abs(dist_pct)
        if enable_u_logic:
            y = ma_segment.values; x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 2)
            if coeffs[0] > 0 and (len(y)*0.3 <= -coeffs[1]/(2*coeffs[0]) <= len(y)*1.1) and (y[-1]-y[-2]) > 0 and coeffs[0] >= min_curvature:
                u_score = (coeffs[0] * 1000) - (abs(dist_pct) * 0.5)
            else: return None
        if not stock.options: return None
        earnings_date = "未知"
        try:
            cal = stock.calendar
            if cal and 'Earnings Date' in cal: earnings_date = cal['Earnings Date'][0].strftime('%m-%d')
        except: pass
        return {
            "代號": symbol, "連結": f"https://finance.yahoo.com/quote/{symbol}", 
            "HV Rank": round(hv_rank, 1), "Week Vol": round(vol_5d, 1),
            "現價": round(df_4h['Close'].iloc[-1], 2), "4H 60MA": round(ma_segment.iloc[-1], 2),
            "乖離率": f"{round(dist_pct, 2)}%", "產業": translate_industry(stock.info.get('industry', 'N/A')),
            "財報日": earnings_date, "題材搜尋": f"https://www.google.com/search?q={symbol}+美股+題材+風險+分析",
            "_sort_score": u_score
        }
    except: return None

# --- 5. 執行邏輯與顯示 (【已修正】加入進度條) ---
if st.button("🚀 啟動 Turbo 掃描", type="primary"):
    st.session_state['scan_results'] = None
    with st.status("正在依據幽靈策略掃描標的...", expanded=True) as status:
        tickers = list(set(get_sp500_tickers() + get_nasdaq100_tickers()))[:scan_limit]
        total_tickers = len(tickers)
        results = []
        
        # 顯示進度條
        progress_bar = st.progress(0)
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(get_ghost_metrics, t, min_volume_threshold): t for t in tickers}
            for future in as_completed(future_to_ticker):
                data = future.result()
                if data: results.append(data)
                
                # 更新進度條
                completed_count += 1
                progress_bar.progress(completed_count / total_tickers)
        
        st.session_state['scan_results'] = results
        status.update(label=f"掃描完成！發現 {len(results)} 檔符合「結冰區」標的。", state="complete", expanded=False)

if 'scan_results' in st.session_state and st.session_state['scan_results']:
    df = pd.DataFrame(st.session_state['scan_results']).sort_values(by="HV Rank", ascending=True)
    st.subheader("📋 符合 Step 1 條件標的清單")
    st.dataframe(df, column_config={
        "代號": st.column_config.LinkColumn("代號", display_text="https://finance\\.yahoo\\.com/quote/(.*)"),
        "連結": None, "_sort_score": None,
        "題材搜尋": st.column_config.LinkColumn("題材", display_text="🔍")
    }, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🕯️ 三週期 K 線檢視")
    selected_option = st.selectbox("選擇股票:", df.apply(lambda x: f"{x['代號']} - {x['產業']}", axis=1).tolist())
    if selected_option:
        plot_interactive_chart(selected_option.split(" - ")[0])
