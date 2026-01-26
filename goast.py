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

# 備份機制
if 'backup' not in st.session_state:
    st.session_state.backup = {
        'scan_limit': 600, 
        'min_vol_m': 10, 
        'dist_threshold': 8.0, 
        'u_sensitivity': 30
    }

# --- 邏輯連動控制中心 ---
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
        st.session_state.u_sensitivity = 240 
    else:
        st.session_state.scan_limit = st.session_state.backup['scan_limit']
        st.session_state.min_vol_m = st.session_state.backup['min_vol_m']
        st.session_state.dist_threshold = st.session_state.backup['dist_threshold']
        st.session_state.u_sensitivity = st.session_state.backup['u_sensitivity']

def handle_spoon_toggle():
    if st.session_state.spoon_strict_key:
        st.session_state.u_sensitivity = 240

def sync_logic_state():
    """總控函數：解決模式連動衝突"""
    is_box_active = st.session_state.get('box_mode_key', False)
    ignition_mode = st.session_state.get('ignition_mode_key', "🚫 不啟用")
    
    if not is_box_active:
        if "週線點火" in ignition_mode:
            if st.session_state.dist_threshold < 50.0:
                st.session_state.backup['dist_threshold'] = st.session_state.dist_threshold
                st.session_state.dist_threshold = 50.0
        else:
            if st.session_state.dist_threshold == 50.0:
                st.session_state.dist_threshold = st.session_state.backup.get('dist_threshold', 8.0)

st.title("👻 幽靈策略掃描器")
st.caption(f"📅 台灣時間：{datetime.now().strftime('%Y-%m-%d %H:%M')} (2026年)")

# --- 2. 核心策略導引區 (完整版) ---
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

# --- 3. 側邊欄設定 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio("市場", ["S&P 500", "NASDAQ 100", "🔥 全火力"], index=2)
scan_limit = st.sidebar.slider("掃描數量", 50, 600, key='scan_limit')

# --- 箱型突破 (霸道模式) ---
st.sidebar.header("📦 箱型突破 (霸道模式)")
enable_box_breakout = st.sidebar.checkbox(
    "✅ 啟動週線橫盤突破 (忽略其他條件)", 
    value=False, 
    key='box_mode_key',
    on_change=sync_logic_state,
    help="啟動此濾網時，將忽略下方的 MA60、乖離率、U型等所有設定，只篩選「盤整突破」的股票。"
)

if enable_box_breakout:
    st.sidebar.warning("⚠️ 霸道模式已啟動：下方其他濾網已暫時失效。")
    
    # 【新增】全自動 VCP 偵測
    enable_full_auto_vcp = st.sidebar.checkbox(
        "🤯 全自動 VCP 偵測 (免設定週數)",
        value=True,
        help="勾選後，系統會自動掃描 10~52 週內的所有可能性，找出符合「波動收縮 (VCP)」且正在突破的型態。"
    )
    
    if not enable_full_auto_vcp:
        box_weeks = st.sidebar.slider("設定盤整週數 (N)", 4, 52, 20, help="股票必須在過去 N 週內橫向整理")
        auto_flag_mode = st.sidebar.checkbox("🤖 自動偵測旗型收斂 (左寬右窄)", value=True)
        if not auto_flag_mode:
            box_tightness = st.sidebar.slider("盤整區間寬度限制 (%)", 10, 50, 25)
        else:
            box_tightness = 100 # 自動旗型下，寬度由邏輯控制
    else:
        st.sidebar.caption("👉 系統將自動尋找最佳的收斂突破週期")
        box_weeks = 52 # 預設給最大，邏輯內部會動態調整
        auto_flag_mode = True # 全自動模式隱含旗型偵測
        box_tightness = 100
else:
    enable_full_auto_vcp = False
    box_weeks = 52
    auto_flag_mode = False
    box_tightness = 25

st.sidebar.divider()

# --- 幽靈戰法設定 ---
st.sidebar.header("📈 幽靈戰法連動")
enable_u_logic = st.sidebar.checkbox("✅ 啟動 4小時 U型戰法連動", value=False, key='u_logic_key', on_change=handle_u_logic_toggle)

enable_spoon_strict = False
spoon_vertex_range = (50, 95)
if enable_u_logic:
    enable_spoon_strict = st.sidebar.checkbox("🥄 嚴格勺子模式", value=True, key='spoon_strict_key', on_change=handle_spoon_toggle)
    if enable_spoon_strict:
        spoon_vertex_range = st.sidebar.slider("🥄 勺子底部位置 (%)", 0, 100, (50, 95), 5)

st.sidebar.header("🛡️ 趨勢與點火")
check_daily_ma60_up = st.sidebar.checkbox("✅ 日線 60MA 向上", value=True)
check_ma60_strong_trend = st.sidebar.checkbox("✅ 週線 MA60 強勢趨勢", value=True)
check_price_above_daily_ma60 = st.sidebar.checkbox("✅ 股價 > 日線 60MA", value=True)

ignition_mode = st.sidebar.radio(
    "動能點火週期:",
    ["🚫 不啟用 (左側佈局)", "⚡ 4H 點火 (短線突破前高)", "🚀 週線點火 (本週突破 OR 上週已突破)"],
    index=0,
    key="ignition_mode_key",
    on_change=sync_logic_state 
)

st.sidebar.header("⚙️ 基礎篩選")
hv_threshold = st.sidebar.slider("HV Rank 門檻", 10, 100, 30)
min_vol_m = st.sidebar.slider("最小日均量 (百萬股)", 1, 100, key='min_vol_m') 
dist_threshold = st.sidebar.slider("距離 4H MA60 範圍 (%)", 0.0, 50.0, key='dist_threshold', step=0.5)

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
                
                shapes = []
                if enable_box_breakout:
                    # 嘗試從 session_state 獲取該股票的偵測週數，若無則用預設
                    detected_weeks = box_weeks
                    # (這裡為了簡化，繪圖時統一畫出最近的 box_weeks，即便自動偵測可能是別的週數)
                    
                    last_n = df.iloc[-(detected_weeks+1):-1]
                    if len(last_n) > 0:
                        box_top = last_n['High'].max()
                        box_bottom = last_n['Low'].min()
                        shapes.append(dict(type="rect", x0=last_n.index[0], y0=box_bottom, x1=last_n.index[-1], y1=box_top, line=dict(color="RoyalBlue"), fillcolor="LightSkyBlue", opacity=0.3))
                
                fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='周K'),
                                 go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=3))])
                
                if shapes: fig.update_layout(shapes=shapes)
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

# --- 6. 核心指標運算 (數據源修復+雙重突破+期權OI+全自動VCP) ---
def get_ghost_metrics(symbol, vol_threshold):
    try:
        stock = yf.Ticker(symbol)
        
        # 統一抓取日線資料 (2年)
        df_daily_2y = stock.history(period="2y", interval="1d")
        
        if len(df_daily_2y) < 250: return None 
        
        # 準備基礎數據
        log_ret = np.log(df_daily_2y['Close'] / df_daily_2y['Close'].shift(1))
        vol_30d = log_ret.rolling(30).std() * np.sqrt(252) * 100
        hv_rank_val = ((vol_30d.iloc[-1] - vol_30d.min()) / (vol_30d.max() - vol_30d.min())) * 100
        
        ma60_4h_val = 0
        dist_pct_val = 0
        
        # --- A. 霸道模式：箱型突破邏輯 ---
        if enable_box_breakout:
            df_wk = df_daily_2y.resample('W').agg({
                'Open': 'first', 
                'High': 'max', 
                'Low': 'min', 
                'Close': 'last', 
                'Volume': 'sum'
            }).dropna()
            
            # 若資料不足基本長度
            if len(df_wk) < 15: return None
            
            avg_vol = df_wk['Volume'].tail(10).mean()
            if avg_vol < vol_threshold * 2: return None 
            
            # --- 全自動 VCP 偵測核心 ---
            # 定義要掃描的候選週期 (優先找長天期)
            if enable_full_auto_vcp:
                candidate_periods = [52, 40, 30, 20, 12]
            else:
                candidate_periods = [box_weeks] # 只檢查使用者設定的單一週期
            
            found_vcp = False
            final_box_weeks = 0
            final_box_high = 0
            final_box_amp = 0
            
            current_week = df_wk.iloc[-1]
            
            for p in candidate_periods:
                if len(df_wk) < p + 2: continue
                
                # 定義區間
                box_start_idx = -(p + 1)
                box_data = df_wk.iloc[box_start_idx:-1]
                
                box_high = box_data['High'].max()
                box_low = box_data['Low'].min()
                
                if box_low == 0: continue
                
                # 自動收斂檢查
                if auto_flag_mode or enable_full_auto_vcp:
                    mid_point = len(box_data) // 2
                    part_old = box_data.iloc[:mid_point]
                    part_recent = box_data.iloc[mid_point:]
                    
                    range_old = part_old['High'].max() - part_old['Low'].min()
                    range_recent = part_recent['High'].max() - part_recent['Low'].min()
                    
                    if range_old == 0: continue
                    
                    # 嚴格收斂條件：近期波動 < 前期波動 * 0.85
                    if range_recent > range_old * 0.85: continue 
                    
                    # 檢查是否突破
                    if current_week['Close'] < box_high * 0.99: continue
                    
                    # 找到符合條件的，鎖定數據並跳出迴圈 (優先回傳長週期)
                    found_vcp = True
                    final_box_weeks = p
                    final_box_high = box_high
                    final_box_amp = (range_recent / box_low) * 100 # 顯示近期收斂幅度
                    break
                else:
                    # 手動模式
                    box_amplitude = (box_high - box_low) / box_low * 100
                    if box_amplitude > box_tightness: continue
                    
                    if current_week['Close'] >= box_high * 0.99:
                        found_vcp = True
                        final_box_weeks = p
                        final_box_high = box_high
                        final_box_amp = box_amplitude
                        break
            
            if not found_vcp: return None
            
            # 補全 4H 數據
            try:
                df_1h = stock.history(period="1y", interval="1h")
                if len(df_1h) > 200:
                    df_4h = df_1h.resample('4h').agg({'Close': 'last'}).dropna()
                    df_4h['MA60'] = df_4h['Close'].rolling(60).mean()
                    ma60_4h_val = df_4h['MA60'].iloc[-1]
                    dist_pct_val = ((df_4h['Close'].iloc[-1] - ma60_4h_val) / ma60_4h_val) * 100
            except: pass

        # --- B. 原本的幽靈策略邏輯 ---
        else:
            df_1h = stock.history(period="1y", interval="1h")
            if len(df_1h) < 240: return None
            df_daily = df_1h.resample('D').agg({'Volume': 'sum', 'Close': 'last'}).dropna()
            df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
            
            if check_daily_ma60_up and df_daily['MA60'].iloc[-1] <= df_daily['MA60'].iloc[-2]: return None
            if df_daily['Volume'].rolling(20).mean().iloc[-1] < vol_threshold: return None
            
            df_wk = None
            if check_ma60_strong_trend or "週線點火" in ignition_mode:
                df_wk = df_daily_2y.resample('W').agg({'Close': 'last', 'High': 'max'}).dropna()
            
            if check_ma60_strong_trend:
                if df_wk is not None and len(df_wk) > 65:
                    df_wk['MA60'] = df_wk['Close'].rolling(60).mean()
                    if not df_wk['MA60'].tail(5).is_monotonic_increasing: return None
                else: return None

            if "週線點火" in ignition_mode:
                if df_wk is not None and len(df_wk) >= 3:
                    curr_price = df_daily_2y['Close'].iloc[-1] 
                    prev_week_high = df_wk['High'].iloc[-2]    
                    prev_week_close = df_wk['Close'].iloc[-2]  
                    prev_2_week_high = df_wk['High'].iloc[-3]  
                    cond1 = curr_price > prev_week_high
                    cond2 = prev_week_close > prev_2_week_high
                    if not (cond1 or cond2): return None
                else: return None

            if check_price_above_daily_ma60 and df_daily['Close'].iloc[-1] < df_daily['MA60'].iloc[-1]: return None
            if hv_rank_val > hv_threshold: return None
            
            df_4h = df_1h.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
            df_4h['MA60'] = df_4h['Close'].rolling(60).mean()
            dist_pct_val = ((df_4h['Close'].iloc[-1] - df_4h['MA60'].iloc[-1]) / df_4h['MA60'].iloc[-1]) * 100
            ma60_4h_val = df_4h['MA60'].iloc[-1]
            
            if abs(dist_pct_val) > dist_threshold: return None
            
            if "4H 點火" in ignition_mode:
                if len(df_4h) < 2: return None
                if df_4h['Close'].iloc[-1] <= df_4h['High'].iloc[-2]: return None
            
            if enable_u_logic:
                y = df_4h['MA60'].tail(u_sensitivity).values; x = np.arange(len(y))
                a, b, c = np.polyfit(x, y, 2)
                vertex_x = -b / (2 * a)
                if a <= 0: return None
                if enable_spoon_strict:
                    min_p, max_p = spoon_vertex_range
                    if not (len(y)*(min_p/100) <= vertex_x <= len(y)*(max_p/100)): return None
                    if y[-1] <= y[-2] or y[0] < y[-1]: return None
                else:
                    if not (len(y)*0.3 <= vertex_x <= len(y)*1.1): return None
                    if y[-1] <= y[-2]: return None
                if a < min_curvature: return None

        # --- 期權數據 (共用) ---
        atm_oi_display = "N/A"
        near_call_max = "N/A"
        near_put_max = "N/A"
        all_call_max = "N/A"
        all_put_max = "N/A"
        
        try:
            opts = stock.options
            if opts:
                chain_near = stock.option_chain(opts[0])
                cur_price = df_daily_2y['Close'].iloc[-1]
                
                closest_idx = (chain_near.calls['strike'] - cur_price).abs().idxmin()
                atm_strike = chain_near.calls.loc[closest_idx, 'strike']
                c_oi = chain_near.calls[chain_near.calls['strike'] == atm_strike]['openInterest'].sum()
                p_oi = chain_near.puts[chain_near.puts['strike'] == atm_strike]['openInterest'].sum()
                atm_oi_display = f"{int(c_oi + p_oi):,}"
                
                if not chain_near.calls.empty:
                    near_call_max = chain_near.calls.loc[chain_near.calls['openInterest'].idxmax(), 'strike']
                if not chain_near.puts.empty:
                    near_put_max = chain_near.puts.loc[chain_near.puts['openInterest'].idxmax(), 'strike']
                
                max_c_oi = 0; max_p_oi = 0
                scan_dates = opts[:6] 
                for d in scan_dates:
                    try:
                        ch = stock.option_chain(d)
                        if not ch.calls.empty:
                            c_max_row = ch.calls.loc[ch.calls['openInterest'].idxmax()]
                            if c_max_row['openInterest'] > max_c_oi:
                                max_c_oi = c_max_row['openInterest']
                                all_call_max = c_max_row['strike']
                        if not ch.puts.empty:
                            p_max_row = ch.puts.loc[ch.puts['openInterest'].idxmax()]
                            if p_max_row['openInterest'] > max_p_oi:
                                max_p_oi = p_max_row['openInterest']
                                all_put_max = p_max_row['strike']
                    except: continue
        except: pass

        earnings_date = "未知"
        cal = stock.calendar
        if cal is not None and 'Earnings Date' in cal:
            earnings_date = cal['Earnings Date'][0].strftime('%m-%d')
            
        week_vol_move = log_ret.tail(5).std() * np.sqrt(5) * 100 if len(log_ret) >= 5 else 0
        move_dollar = df_daily_2y['Close'].iloc[-1] * (week_vol_move / 100)
        
        # 顯示處理
        if enable_box_breakout:
            box_str = f"箱頂 {round(final_box_high, 2)}"
            box_amp_str = f"VCP{final_box_weeks}W:{round(final_box_amp, 2)}%" 
        else:
            box_str = f"±{round(move_dollar, 2)}"
            box_amp_str = round(week_vol_move, 2)

        return {
            "代號": symbol, 
            "HV Rank": round(hv_rank_val, 1), 
            "週波動%": box_amp_str, # 顯示 VCP 週期與收斂度
            "預期變動$": box_str, 
            "現價": round(df_daily_2y['Close'].iloc[-1], 2),
            "4H 60MA": round(ma60_4h_val, 2) if ma60_4h_val != 0 else "N/A",
            "4H MA60 乖離率": f"{round(dist_pct_val, 2)}%" if ma60_4h_val != 0 else "N/A",
            "價平OI": atm_oi_display,
            "近Call大量": near_call_max,
            "近Put大量": near_put_max,
            "全Call大量": all_call_max,
            "全Put大量": all_put_max,
            "產業": translate_industry(stock.info.get('industry', 'N/A')),
            "下次財報": earnings_date, 
            "題材搜尋": f"https://www.google.com/search?q={symbol}+題材+風險", 
            "_sort_score": 99999 if enable_box_breakout else -abs(dist_pct_val)
        }
    except: return None

# --- 7. 抓取代號 ---
@st.cache_data(ttl=3600)
def get_tickers_robust(choice):
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers = []
    try: 
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[0]
        tickers.extend(df[df.columns[0]].tolist())
    except: pass
    try: 
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
    
    status_text = "🔍 掃描中 (霸道模式)..." if enable_box_breakout else "🔍 掃描中..."
    
    with st.status(status_text, expanded=True) as status:
        tickers = get_tickers_robust(market_choice)[:scan_limit]
        total_tickers = len(tickers)
        
        status.write(f"✅ 已獲得 {total_tickers} 檔代號，開始技術面過濾...")
        
        results = []; count = 0
        progress = st.progress(0)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(get_ghost_metrics, t, min_volume_threshold): t for t in tickers}
            for future in as_completed(future_to_ticker):
                data = future.result(); count += 1
                progress.progress(count / total_tickers if total_tickers > 0 else 0)
                if data: results.append(data)
        st.session_state['scan_results'] = results
        status.update(label=f"完成！共 {len(results)} 檔。", state="complete", expanded=False)

if 'scan_results' in st.session_state and st.session_state['scan_results']:
    df = pd.DataFrame(st.session_state['scan_results']).sort_values(by="HV Rank", ascending=True)
    
    df_display = df.copy()
    df_display["代號"] = df_display["代號"].apply(lambda x: f"https://finance.yahoo.com/quote/{x}/key-statistics")

    st.subheader("📋 策略篩選列表")
    if enable_box_breakout:
        st.caption(f"🔥 目前顯示：符合【連續 {box_weeks} 週橫盤 + 本週突破】之強勢股")
    
    st.dataframe(
        df_display,
        column_config={
            "代號": st.column_config.LinkColumn("代號 (點我跳轉)", display_text="https://finance\\.yahoo\\.com/quote/(.*?)/key-statistics"),
            "題材搜尋": st.column_config.LinkColumn("題材與風險", display_text="🔍 查詢"),
            "_sort_score": None
        },
        hide_index=True, use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("🕯️ K 線檢視")
    
    options = df.apply(lambda x: f"{x['代號']} - {x['產業']}", axis=1).tolist()

    if options:
        default_option = options[0]
        
        selected_pill = st.pills(
            "👉 請點擊標的 (不會跳出鍵盤)",
            options,
            default=default_option,
            selection_mode="single",
            key="pills_selector"
        )
        
        if selected_pill:
            target = selected_pill.split(" - ")[0]
            st.caption(f"目前檢視: {target}")
            plot_interactive_chart(target)
        else:
            st.info("請點選上方標籤以查看 K 線")
    else:
        st.write("查無符合條件標的")
