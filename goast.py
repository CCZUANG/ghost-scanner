import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="幽靈策略掃描器 (新聞透視版)", page_icon="👻", layout="wide")

st.title("👻 幽靈策略掃描器 (新聞透視版)")
st.write("""
**策略目標**：以 **HV 低波動** 排序，尋找 **日線多頭 + 4H U型** 的標的，並提供 **中文產業** 與 **最新新聞**。
""")

# --- 2. 側邊欄：參數設定區 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio(
    "選擇掃描市場", 
    ["S&P 500 (大型股)", "NASDAQ 100 (科技股)", "🔥 全火力 (兩者全掃)"],
    index=2
)
scan_limit = st.sidebar.slider("掃描數量 (前 N 大)", 50, 600, 200)

# --- 日線趨勢濾網 ---
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

# --- 產業翻譯字典 ---
INDUSTRY_MAP = {
    "Technology": "科技",
    "Financial Services": "金融",
    "Healthcare": "醫療保健",
    "Consumer Cyclical": "非必需消費",
    "Consumer Defensive": "必需消費",
    "Energy": "能源",
    "Industrials": "工業",
    "Communication Services": "通訊服務",
    "Utilities": "公用事業",
    "Real Estate": "房地產",
    "Basic Materials": "原物料",
    "Semiconductors": "半導體",
    "Software - Infrastructure": "軟體基礎設施",
    "Software - Application": "應用軟體",
    "Internet Content & Information": "網路內容",
    "Banks - Diversified": "銀行",
    "Credit Services": "信貸服務",
    "Aerospace & Defense": "航太軍工",
    "Auto Manufacturers": "汽車製造"
}

def translate_industry(eng_industry):
    if not eng_industry or eng_industry == "N/A":
        return "未知"
    # 先嘗試直接對應
    if eng_industry in INDUSTRY_MAP:
        return INDUSTRY_MAP[eng_industry]
    # 如果找不到，嘗試部分關鍵字匹配
    for key, value in INDUSTRY_MAP.items():
        if key in eng_industry:
            return value
    return eng_industry # 真的翻不出來就顯示原文

# --- 3. 核心函數 ---

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(response.text))[0]
        tickers = df['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except:
        return []

@st.cache_data(ttl=3600)
def get_nasdaq100_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=headers)
        dfs = pd.read_html(StringIO(response.text))
        for df in dfs:
            if 'Ticker' in df.columns:
                tickers = df['Ticker'].tolist()
                return [t.replace('.', '-') for t in tickers]
            elif 'Symbol' in df.columns:
                tickers = df['Symbol'].tolist()
                return [t.replace('.', '-') for t in tickers]
        return []
    except:
        return []

def get_combined_tickers(choice, limit):
    sp500 = []
    nasdaq = []
    
    if "S&P" in choice or "全火力" in choice:
        sp500 = get_sp500_tickers()
    
    if "NASDAQ" in choice or "全火力" in choice:
        nasdaq = get_nasdaq100_tickers()
    
    combined = list(set(sp500 + nasdaq))
    
    if not combined:
        return ['TSM', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'PLTR', 'LUNR', 'COIN', 'MSTR', 'QQQ', 'SPY']
    
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
        
        if not (len_window * 0.3 <= vertex_x <= len_window * 1.1):
            return False, a
            
        current_slope = y[-1] - y[-2]
        if current_slope <= 0: return False, a

        return True, a
    except:
        return False, 0

def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol)
        df_1h = stock.history(period="6mo", interval="1h")
        if len(df_1h) < 240: return None

        # --- A. 日線級別處理 ---
        df_daily_synth = df_1h.resample('D').agg({
            'Volume': 'sum',
            'Close': 'last'
        }).dropna()
        
        df_daily_synth['MA60'] = df_daily_synth['Close'].rolling(window=60).mean()
        
        if len(df_daily_synth) < 60: return None
        
        daily_ma60_now = df_daily_synth['MA60'].iloc[-1]
        daily_ma60_prev = df_daily_synth['MA60'].iloc[-2]
        current_price_daily = df_daily_synth['Close'].iloc[-1]

        if check_daily_ma60_up and daily_ma60_now <= daily_ma60_prev: return None
        if check_price_above_daily_ma60 and current_price_daily < daily_ma60_now: return None

        avg_volume = df_daily_synth['Volume'].rolling(window=20).mean().iloc[-1]
        if avg_volume < vol_threshold: return None

        close_daily = df_daily_synth['Close']
        log_ret = np.log(close_daily / close_daily.shift(1))
        vol_30d = log_ret.rolling(window=30).std() * np.sqrt(252) * 100
        
        current_hv = vol_30d.iloc[-1]
        min_hv = vol_30d.min()
        max_hv = vol_30d.max()
        if max_hv == min_hv: return None
        hv_rank = ((current_hv - min_hv) / (max_hv - min_hv)) * 100
        
        if hv_rank > hv_threshold: return None

        # --- B. 4小時級別處理 ---
        df_4h = df_1h.resample('4h').agg({
            'Close': 'last', 
            'Volume': 'sum'
        }).dropna()
        
        if len(df_4h) < 60: return None

        df_4h['MA60'] = df_4h['Close'].rolling(window=60).mean()
        ma_segment = df_4h['MA60'].iloc[-u_sensitivity:]
        if ma_segment.isnull().values.any() or len(ma_segment) < u_sensitivity: return None
        
        current_price_4h = df_4h['Close'].iloc[-1]
        ma60_now_4h = ma_segment.iloc[-1]
        dist_pct = ((current_price_4h - ma60_now_4h) / ma60_now_4h) * 100

        if abs(dist_pct) > dist_threshold: return None 
        
        # --- C. U 型檢測 ---
        u_score = 0
        curvature = 0

        if enable_u_logic:
            is_u_shape, curv = analyze_u_shape(ma_segment)
            if not is_u_shape: return None
            if curv < min_curvature: return None
            curvature = curv
            u_score = (curvature * 1000) - (abs(dist_pct) * 0.5)
        else:
            u_score = -abs(dist_pct)

        # --- D. 期權檢查 ---
        try:
            if not stock.options: return None
        except:
            return None

        # --- E. 財報、產業與新聞 (資訊豐富化) ---
        industry_tw = "未知"
        news_title = "無新聞"
        news_link = None
        earnings_date_str = "未知"

        try:
            # 1. 產業資訊
            info = stock.info
            raw_industry = info.get('industry', info.get('sector', 'N/A'))
            industry_tw = translate_industry(raw_industry)
            
            # 2. 財報日期
            cal = stock.calendar
            if cal and isinstance(cal, dict) and 'Earnings Date' in cal:
                earnings_date_str = cal['Earnings Date'][0].strftime('%m-%d')
            elif cal and isinstance(cal, dict) and 'Earnings High' in cal:
                 earnings_date_str = cal['Earnings High'][0].strftime('%m-%d')
            
            # 3. 最新新聞 (抓第一則)
            news_list = stock.news
            if news_list and len(news_list) > 0:
                latest_news = news_list[0]
                news_title = latest_news.get('title', '無標題')
                news_link = latest_news.get('link', None)
        except:
            pass

        return {
            "代號": symbol,
            "HV Rank": round(hv_rank, 1),
            "現價": round(current_price_4h, 2),
            "4H 60MA": round(ma60_now_4h, 2),
            "乖離率": f"{round(dist_pct, 2)}%",
            "產業": industry_tw,
            "財報日": earnings_date_str,
            "最新新聞": news_title,
            "新聞連結": news_link, # 隱藏欄位，用於生成連結
            "_sort_score": u_score,
            "_dist_raw": abs(dist_pct)
        }
    except:
        return None

# --- 4. 主程式執行邏輯 ---

if st.button("🚀 啟動 Turbo 掃描", type="primary"):
    status_text = f"正在下載 {market_choice} 清單..."
    progress_bar = st.progress(0)
    
    with st.status(status_text, expanded=True) as status:
        target_tickers = get_combined_tickers(market_choice, scan_limit)
        
        status.write(f"🔥 Turbo 模式啟動！ (核心數: {max_workers})")
        status.write(f"🔍 目標: {len(target_tickers)} 檔 | 正在抓取財報、新聞與中文產業...")
        
        results = []
        completed_count = 0
        total_count = len(target_tickers)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(get_ghost_metrics, t, min_volume_threshold): t 
                for t in target_tickers
            }
            
            for future in as_completed(future_to_ticker):
                data = future.result()
                if data:
                    results.append(data)
                
                completed_count += 1
                progress_bar.progress(completed_count / total_count)
            
        status.update(label=f"掃描完成！共發現 {len(results)} 檔。", state="complete", expanded=False)

    if results:
        df_results = pd.DataFrame(results)
        
        # 【修改】依照使用者要求：HV Rank 由低到高排序 (越冷越好)
        df_results = df_results.sort_values(by="HV Rank", ascending=True)
        
        st.success(f"🎯 發現 {len(df_results)} 檔優質標的！(已按 HV 低波動排序)")
        
        column_config = {
            "HV Rank": st.column_config.NumberColumn("HV波動 (低優先)", format="%.1f"),
            "現價": st.column_config.NumberColumn(format="$%.2f"),
            "4H 60MA": st.column_config.NumberColumn("4H 季線", format="$%.2f"),
            "乖離率": st.column_config.TextColumn("距離均線"),
            "產業": st.column_config.TextColumn("產業類別"),
            "財報日": st.column_config.TextColumn("下季財報"),
            "最新新聞": st.column_config.LinkColumn(
                "最新新聞 (點擊閱讀)", 
                display_text="最新新聞" # 讓表格顯示標題文字，點擊跳轉
            ),
            # 隱藏輔助欄位
            "新聞連結": None,
            "_sort_score": None,
            "_dist_raw": None
        }

        # 處理新聞連結：將標題與連結合併給 LinkColumn 使用
        # Streamlit 的 LinkColumn 需要一個完整的 URL，
        # 這裡我們做一個小技巧：把 '最新新聞' 欄位直接變成連結文字，
        # 或者更簡單地，將 '新聞連結' 設為 LinkColumn，並顯示 '最新新聞' 的標題
        
        # 修正：Streamlit 的 LinkColumn 目前比較適合顯示固定文字或 URL
        # 我們把 '最新新聞' 欄位原本的標題文字換成 (URL, 標題) 的 tuple 結構是無效的
        # 最好的方式是：把 '新聞連結' 設為 LinkColumn，display_text 使用 '最新新聞' 欄位的內容
        
        st.dataframe(
            df_results,
            column_config={
                "HV Rank": st.column_config.NumberColumn("HV波動 (低優先)", format="%.1f"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "4H 60MA": st.column_config.NumberColumn("4H 季線", format="$%.2f"),
                "乖離率": st.column_config.TextColumn("距離均線"),
                "產業": st.column_config.TextColumn("產業類別"),
                "財報日": st.column_config.TextColumn("下季財報"),
                "最新新聞": st.column_config.Column("新聞標題"), # 純文字顯示標題
                "新聞連結": st.column_config.LinkColumn("閱讀", display_text="Go 🔗"), # 按鈕跳轉
                "_sort_score": None,
                "_dist_raw": None
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("⚠️ 沒掃到符合條件的股票。\n建議：\n1. 放寬「HV Rank 門檻」\n2. 嘗試取消勾選「日線 60MA 向上」")
