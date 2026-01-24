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
st.set_page_config(page_title="幽靈策略掃描器 (防手滑版)", page_icon="👻", layout="wide")

st.title("👻 幽靈策略掃描器 (防手滑版)")
st.write("""
**策略目標**：鎖定 **日線多頭 + 4H U型**，圖表已鎖定防誤觸，點擊代號可開外部連結。
""")

# --- 2. 側邊欄：參數設定區 ---
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
hv_threshold = st.sidebar.slider("HV Rank 門檻 (越低越好)", 10, 100, 65)
min_vol_m = st.sidebar.slider("最小日均量 (百萬股)", 1, 20, 3) 
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

# --- 3. 輔助與核心函數 ---

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

# --- 改進版繪圖函數 (防誤觸設定) ---
def plot_interactive_chart(symbol):
    stock = yf.Ticker(symbol)
    
    # 【優化】簡化名稱，避免在窄螢幕下分頁標籤被擠掉
    tab1, tab2, tab3 = st.tabs(["🗓️ 周線", "📅 日線", "⏱️ 4H"])
    
    # 共用佈局
    layout_common = dict(
        xaxis_rangeslider_visible=False,
        height=600,  
        margin=dict(l=10, r=10, t=50, b=50), 
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5
        ),
        # 【關鍵修正】禁止預設的拖曳 (防手滑)
        dragmode=False, 
    )

    # 定義標題樣式
    def get_title_config(text):
        return dict(text=text, x=0.02, xanchor='left', font=dict(size=16))

    # 定義圖表設定 (禁止滾輪縮放)
    config_common = {
        'scrollZoom': False,       # 禁止滑鼠滾輪縮放
        'displayModeBar': True,    # 顯示工具列 (需要縮放時可以自己點)
        'displaylogo': False
    }

    # --- Tab 1: 周線圖 ---
    with tab1:
        try:
            df_w = stock.history(period="5y", interval="1wk")
            if len(df_w) < 60:
                st.warning("周線數據不足")
            else:
                df_w['MA20'] = df_w['Close'].rolling(window=20).mean()
                df_w['MA60'] = df_w['Close'].rolling(window=60).mean()
                df_w_view = df_w.iloc[-100:]

                fig_w = go.Figure()
                fig_w.add_trace(go.Candlestick(
                    x=df_w_view.index, open=df_w_view['Open'], high=df_w_view['High'],
                    low=df_w_view['Low'], close=df_w_view['Close'], name='周K'
                ))
                fig_w.add_trace(go.Scatter(
                    x=df_w_view.index, y=df_w_view['MA20'], mode='lines', name='MA20',
                    line=dict(color='royalblue', width=1)
                ))
                fig_w.add_trace(go.Scatter(
                    x=df_w_view.index, y=df_w_view['MA60'], mode='lines', name='MA60',
                    line=dict(color='orange', width=3)
                ))
                
                fig_w.update_layout(title=get_title_config(f"{symbol} 周線"), yaxis_title="股價", **layout_common)
                st.plotly_chart(fig_w, use_container_width=True, config=config_common)
        except Exception as e:
            st.error(f"周線圖錯誤: {e}")

    # --- Tab 2: 日線圖 ---
    with tab2:
        try:
            df_d = stock.history(period="1y")
            if len(df_d) < 60:
                st.warning("日線數據不足")
            else:
                df_d['MA20'] = df_d['Close'].rolling(window=20).mean()
                df_d['MA60'] = df_d['Close'].rolling(window=60).mean()
                df_d_view = df_d.iloc[-150:] 

                fig_d = go.Figure()
                fig_d.add_trace(go.Candlestick(
                    x=df_d_view.index, open=df_d_view['Open'], high=df_d_view['High'],
                    low=df_d_view['Low'], close=df_d_view['Close'], name='日K'
                ))
                fig_d.add_trace(go.Scatter(
                    x=df_d_view.index, y=df_d_view['MA20'], mode='lines', name='MA20',
                    line=dict(color='royalblue', width=1)
                ))
                fig_d.add_trace(go.Scatter(
                    x=df_d_view.index, y=df_d_view['MA60'], mode='lines', name='MA60',
                    line=dict(color='orange', width=3)
                ))
                
                fig_d.update_layout(title=get_title_config(f"{symbol} 日線"), yaxis_title="股價", **layout_common)
                fig_d.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                st.plotly_chart(fig_d, use_container_width=True, config=config_common)
        except Exception as e:
            st.error(f"日線圖錯誤: {e}")

    # --- Tab 3: 4小時圖 ---
    with tab3:
        try:
            df_1h = stock.history(period="6mo", interval="1h")
            if len(df_1h) < 100:
                st.warning("4H 數據不足")
            else:
                df_4h = df_1h.resample('4h').agg({
                    'Open': 'first', 'High': 'max', 
                    'Low': 'min', 'Close': 'last'
                }).dropna()

                df_4h['MA20'] = df_4h['Close'].rolling(window=20).mean()
                df_4h['MA60'] = df_4h['Close'].rolling(window=60).mean()
                df_4h_view = df_4h.iloc[-80:] 

                fig_4h = go.Figure()
                fig_4h.add_trace(go.Candlestick(
                    x=df_4h_view.index, 
                    open=df_4h_view['Open'], high=df_4h_view['High'],
                    low=df_4h_view['Low'], close=df_4h_view['Close'], 
                    name='4H K'
                ))
                fig_4h.add_trace(go.Scatter(
                    x=df_4h_view.index, y=df_4h_view['MA60'], 
                    mode='lines', name='MA60',
                    line=dict(color='orange', width=3)
                ))
                
                fig_4h.update_layout(title=get_title_config(f"{symbol} 4小時圖"), yaxis_title="股價", **layout_common)
                fig_4h.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                st.plotly_chart(fig_4h, use_container_width=True, config=config_common)
                
        except Exception as e:
            st.error(f"4H 圖錯誤: {e}")

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(response.text))[0]
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except: return []

@st.cache_data(ttl=3600)
def get_nasdaq100_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=headers)
        dfs = pd.read_html(StringIO(response.text))
        for df in dfs:
            if 'Ticker' in df.columns: return [t.replace('.', '-') for t in df['Ticker'].tolist()]
            elif 'Symbol' in df.columns: return [t.replace('.', '-') for t in df['Symbol'].tolist()]
        return []
    except: return []

def get_combined_tickers(choice, limit):
    sp500 = []
    nasdaq = []
    if "S&P" in choice or "全火力" in choice: sp500 = get_sp500_tickers()
    if "NASDAQ" in choice or "全火力" in choice: nasdaq = get_nasdaq100_tickers()
    combined = list(set(sp500 + nasdaq))
    if not combined: return ['TSM', 'NVDA', 'AAPL', 'MSFT', 'AMD', 'PLTR']
    return combined[:limit]

def analyze_u_shape(ma_series):
    try:
        y = ma_series.values
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs
        if a <= 0: return False, 0
        vertex_x = -b / (2 * a)
        len_window = len(y)
        if not (len_window * 0.3 <= vertex_x <= len_window * 1.1): return False, a
        if (y[-1] - y[-2]) <= 0: return False, a
        return True, a
    except: return False, 0

def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol)
        df_1h = stock.history(period="6mo", interval="1h")
        if len(df_1h) < 240: return None

        df_daily_synth = df_1h.resample('D').agg({'Volume': 'sum', 'Close': 'last'}).dropna()
        df_daily_synth['MA60'] = df_daily_synth['Close'].rolling(window=60).mean()
        if len(df_daily_synth) < 60: return None
        
        daily_ma60_now = df_daily_synth['MA60'].iloc[-1]
        daily_ma60_prev = df_daily_synth['MA60'].iloc[-2]
        current_price_daily = df_daily_synth['Close'].iloc[-1]

        if check_daily_ma60_up and daily_ma60_now <= daily_ma60_prev: return None
        if check_price_above_daily_ma60 and current_price_daily < daily_ma60_now: return None

        if df_daily_synth['Volume'].rolling(window=20).mean().iloc[-1] < vol_threshold: return None

        close_daily = df_daily_synth['Close']
        log_ret = np.log(close_daily / close_daily.shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        current_hv = vol_30d.iloc[-1]
        min_hv, max_hv = vol_30d.min(), vol_30d.max()
        if max_hv == min_hv: return None
        hv_rank = ((current_hv - min_hv) / (max_hv - min_hv)) * 100
        if hv_rank > hv_threshold: return None

        df_4h = df_1h.resample('4h').agg({'Close': 'last', 'Volume': 'sum'}).dropna()
        if len(df_4h) < 60: return None
        df_4h['MA60'] = df_4h['Close'].rolling(window=60).mean()
        ma_segment = df_4h['MA60'].iloc[-u_sensitivity:]
        if len(ma_segment) < u_sensitivity: return None
        
        current_price_4h = df_4h['Close'].iloc[-1]
        ma60_now_4h = ma_segment.iloc[-1]
        dist_pct = ((current_price_4h - ma60_now_4h) / ma60_now_4h) * 100
        if abs(dist_pct) > dist_threshold: return None 
        
        u_score, curvature = -abs(dist_pct), 0
        if enable_u_logic:
            is_u, curv = analyze_u_shape(ma_segment)
            if not is_u or curv < min_curvature: return None
            curvature = curv
            u_score = (curvature * 1000) - (abs(dist_pct) * 0.5)

        try:
            if not stock.options: return None
        except: return None

        industry_tw, earnings_date_str = "未知", "未知"
        try:
            info = stock.info
            industry_tw = translate_industry(info.get('industry', info.get('sector', 'N/A')))
            cal = stock.calendar
            if cal and isinstance(cal, dict):
                if 'Earnings Date' in cal: earnings_date_str = cal['Earnings Date'][0].strftime('%m-%d')
                elif 'Earnings High' in cal: earnings_date_str = cal['Earnings High'][0].strftime('%m-%d')
        except: pass

        return {
            "代號": symbol,
            "連結": f"https://finance.yahoo.com/quote/{symbol}", 
            "HV Rank": round(hv_rank, 1),
            "現價": round(current_price_4h, 2),
            "4H 60MA": round(ma60_now_4h, 2),
            "乖離率": f"{round(dist_pct, 2)}%",
            "產業": industry_tw,
            "財報日": earnings_date_str,
            "題材搜尋": f"https://www.google.com/search?q={symbol}+美股+題材+風險+分析",
            "_sort_score": u_score
        }
    except: return None

# --- 4. 主程式執行邏輯 ---

if st.button("🚀 啟動 Turbo 掃描", type="primary"):
    st.session_state['scan_results'] = None
    status_text = f"正在下載 {market_choice} 清單..."
    progress_bar = st.progress(0)
    
    with st.status(status_text, expanded=True) as status:
        target_tickers = get_combined_tickers(market_choice, scan_limit)
        status.write(f"🔥 Turbo 模式啟動！目標: {len(target_tickers)} 檔")
        
        results = []
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(get_ghost_metrics, t, min_volume_threshold): t for t in target_tickers}
            for future in as_completed(future_to_ticker):
                data = future.result()
                if data: results.append(data)
                completed_count += 1
                progress_bar.progress(completed_count / len(target_tickers))
        
        status.update(label=f"掃描完成！共發現 {len(results)} 檔。", state="complete", expanded=False)
        st.session_state['scan_results'] = results

# --- 5. 顯示結果 ---

if 'scan_results' in st.session_state and st.session_state['scan_results']:
    df_results = pd.DataFrame(st.session_state['scan_results'])
    df_results = df_results.sort_values(by="HV Rank", ascending=True)
    
    st.success(f"🎯 發現 {len(df_results)} 檔優質標的！")

    # 【佈局調整】給圖表更多空間 (55 vs 45) -> 改為 4 vs 5
    col1, col2 = st.columns([4, 5])
    
    with col1:
        st.subheader("📋 掃描結果列表")
        column_config = {
            "代號": st.column_config.LinkColumn(
                "代號", 
                display_text="https://finance\\.yahoo\\.com/quote/(.*)", 
                help="點擊開 Yahoo"
            ),
            "連結": None, 
            "HV Rank": st.column_config.NumberColumn("HV", format="%.0f"), # 簡化標題
            "現價": st.column_config.NumberColumn(format="$%.2f"),
            "4H 60MA": st.column_config.NumberColumn("季線", format="$%.2f"), # 簡化標題
            "乖離率": st.column_config.TextColumn("乖離"), # 簡化標題
            "產業": st.column_config.TextColumn("產業"),
            "財報日": st.column_config.TextColumn("財報"),
            "題材搜尋": st.column_config.LinkColumn("題材", display_text="🔍"),
            "_sort_score": None
        }
        
        df_display = df_results.copy()
        df_display["代號"] = df_display["連結"] 
        
        st.dataframe(
            df_display,
            column_config=column_config,
            hide_index=True,
            use_container_width=True
        )

    with col2:
        st.subheader("🕯️ K線檢視器 (防誤觸)")
        st.info("👇 選擇股票後，可點選分頁切換時框")
        select_options = df_results.apply(lambda x: f"{x['代號'].split('/')[-1]} - {x['產業']}", axis=1).tolist()
        selected_option = st.selectbox("選擇股票:", select_options)
        
        if selected_option:
            selected_symbol = selected_option.split(" - ")[0]
            plot_interactive_chart(selected_symbol)
            st.markdown(f"**提示：** 圖表已鎖定縮放。如需放大，請點擊圖表右上角工具列。")
