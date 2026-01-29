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

# --- 邏輯連動 ---
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
    # 確保策略互斥或共存的邏輯
    pass

st.title("👻 幽靈策略掃描器")
st.caption(f"📅 台灣時間：{datetime.now().strftime('%Y-%m-%d %H:%M')} (2026年)")

# --- 2. 核心策略導引區 (美化版：卡片式設計回歸) ---
with st.expander("📖 幽靈策略：動態蝴蝶演化三部曲 (點擊展開)", expanded=False):
    c1, c2, c3 = st.columns(3)
    
    with c1:
        with st.container(border=True):
            st.markdown("### 🏁 Step 1: 試探")
            st.caption("建立多頭價差 (Bull Call Spread)")
            st.info("**🚀 啟動**：突破壓力 / 回測支撐")
            st.markdown("**🛒 動作**：\n- Buy 低價 Call\n- Sell 高價 Call")
            st.success("**✅ 成功**：Delta 隨股價增加")
            st.error("**❌ 失敗**：橫盤 > 2天 或 跌破支撐")

    with c2:
        with st.container(border=True):
            st.markdown("### ❄️ Step 2: 加碼")
            st.caption("動能爆發 (Gamma Scalping)")
            st.info("**🚀 啟動**：價差浮盈 + **IV 膨脹**")
            st.markdown("**🛒 動作**：\n- 加買 更高階 Call\n- (水結成冰戰法)")
            st.success("**✅ 成功**：部位價值隨波動暴增")
            st.error("**❌ 失敗**：動能消失 / IV 萎縮")

    with c3:
        with st.container(border=True):
            st.markdown("### 🦋 Step 3: 鎖利")
            st.caption("轉化蝴蝶 (Butterfly)")
            st.info("**🚀 啟動**：過熱 / 乖離率過大")
            st.markdown("**🛒 動作**：\n- 賣出 中間價 Call\n- 形成 (+1 / -2 / +1) 結構")
            st.success("**✅ 成功**：鎖定 **負成本** (無風險)")
            st.error("**❌ 失敗**：股價遠超最高履約價")
    
    st.warning("💡 **核心心法**：Step 2 的關鍵是 **「IV (隱含波動率) 的擴張」**。只有當市場瘋狂追價時，才值得加碼。")

st.markdown("---")

# --- 3. 側邊欄 ---
st.sidebar.header("🎯 1. 市場設定")
col_m1, col_m2 = st.sidebar.columns([1.5, 1])
with col_m1:
    market_choice = st.radio("選擇市場", ["S&P 500", "NASDAQ 100", "🔥 全火力"], index=2, label_visibility="collapsed")
with col_m2:
    scan_limit = st.number_input("掃描數", min_value=10, max_value=600, step=50, key='scan_limit')

debug_mode = st.sidebar.checkbox("🐞 除錯模式", value=True, help="顯示詳細的失敗原因表格")

st.sidebar.divider()

st.sidebar.subheader("🧠 2. 核心策略")
settings = {}

# A. 霸道模式
enable_box_breakout = st.sidebar.checkbox("📦 啟動：箱型/VCP 霸道模式", value=False, key='box_mode_key', on_change=sync_logic_state)
settings['enable_box_breakout'] = enable_box_breakout

if enable_box_breakout:
    with st.sidebar.container(border=True):
        enable_full_auto_vcp = st.checkbox("🤯 全自動 VCP 偵測", value=True)
        settings['enable_full_auto_vcp'] = enable_full_auto_vcp
        if not enable_full_auto_vcp:
            box_weeks = st.slider("設定盤整週數 (N)", 4, 52, 20)
            settings['box_weeks'] = box_weeks
            auto_flag_mode = st.checkbox("🤖 自動偵測旗型", value=True)
            settings['auto_flag_mode'] = auto_flag_mode
            settings['box_tightness'] = 100 if auto_flag_mode else st.slider("寬度限制 (%)", 10, 50, 25)
        else:
            st.caption("👉 系統將自動尋找最佳週期 (12W~52W)")
            settings['box_weeks'] = 52; settings['auto_flag_mode'] = True; settings['box_tightness'] = 100
else:
    settings['enable_full_auto_vcp'] = False; settings['box_weeks'] = 52; settings['auto_flag_mode'] = False; settings['box_tightness'] = 25

# B. 落水狗反彈模式
enable_reversal_mode = st.sidebar.checkbox("🌊 啟動：落水狗反彈 (MA60下彎 + MA5金叉)", value=False, key='reversal_mode_key')
settings['enable_reversal_mode'] = enable_reversal_mode

# C. 趨勢特快車模式
enable_trend_mode = st.sidebar.checkbox("🚀 啟動：趨勢特快車 (均線多頭+發散噴出)", value=False, key='trend_mode_key')
settings['enable_trend_mode'] = enable_trend_mode

# D. 幽靈模式
enable_u_logic = st.sidebar.checkbox("👻 啟動：U型/勺子 幽靈戰法", value=False, key='u_logic_key', on_change=handle_u_logic_toggle)
settings['enable_u_logic'] = enable_u_logic

if enable_u_logic:
    with st.sidebar.container(border=True):
        st.checkbox("🥄 嚴格勺子模式", value=True, key='spoon_strict_key', on_change=handle_spoon_toggle)
        settings['spoon_strict'] = st.session_state.spoon_strict_key
        settings['spoon_vertex_range'] = st.slider("底部位置 (%)", 0, 100, (50, 95), 5)
        st.markdown("---")
        settings['u_sensitivity'] = st.slider("U型敏感度", 20, 240, key='u_sensitivity')
        settings['min_curvature'] = st.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")
else: 
    settings['spoon_strict'] = False; settings['spoon_vertex_range'] = (50, 95); settings['u_sensitivity'] = 30; settings['min_curvature'] = 0.003

st.sidebar.divider()

st.sidebar.subheader("🛡️ 3. 趨勢與濾網")
default_ma60_up = True
if enable_reversal_mode or enable_trend_mode:
    default_ma60_up = False

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    settings['check_daily_ma60_up'] = st.checkbox("日60MA向上", value=default_ma60_up, disabled=(enable_reversal_mode or enable_trend_mode), help="特殊策略模式下自動由策略內部控管")
    settings['check_price_above_daily_ma60'] = st.checkbox("股價 > 日MA", value=True)
with col_t2:
    settings['check_ma60_strong_trend'] = st.checkbox("週趨勢強勢", value=False if (enable_reversal_mode or enable_trend_mode) else True)

ignition_mode = st.sidebar.radio("動能點火週期:", ["🚫 不啟用", "⚡ 4H 點火", "🚀 週線點火"], index=0, horizontal=True, key="ignition_mode_key", on_change=sync_logic_state)
settings['ignition_mode'] = ignition_mode

with st.sidebar.expander("⚙️ 進階參數", expanded=False):
    settings['hv_threshold'] = st.slider("HV Rank 上限", 10, 100, 30)
    min_vol_m = st.slider("最小日均量 (百萬股)", 1, 100, key='min_vol_m') 
    dist_threshold = st.slider("距離 4H MA60 容許範圍 (%)", 0.0, 50.0, step=0.5, key='dist_threshold')
    settings['dist_threshold'] = dist_threshold
    max_workers = st.slider("🚀 平行運算核心數", 1, 32, 16)

# --- 4. 產業翻譯 ---
def translate_industry(eng):
    if not eng: return "未知"
    mp = {"technology":"科技","software":"軟體","financial":"金融","healthcare":"醫療","energy":"能源","industrials":"工業","real estate":"房產"}
    for k,v in mp.items():
        if k in eng.lower(): return v
    return eng

# --- 5. 繪圖函數 (全線圖優化：修復斷層與拖曳) ---
def plot_interactive_chart(symbol, call_wall, put_wall, vcp_weeks=0, *args, **kwargs):
    stock = yf.Ticker(symbol)
    tab1, tab2, tab3 = st.tabs(["🗓️ 周線", "📅 日線", "⏱️ 4H"])
    
    # 共同 Layout：解決手機顯示問題
    layout_common = dict(
        xaxis_rangeslider_visible=False, 
        height=500, 
        margin=dict(l=0, r=60, t=30, b=20), 
        legend=dict(orientation="h", y=0.99, x=0.01, bgcolor="rgba(0,0,0,0)"), 
        dragmode='pan'
    )
    
    box_shapes = []
    is_box_mode = st.session_state.get('box_mode_key', False)
    
    # 標籤分流：Call上 Put下
    def get_wall_shapes_annotations(cw, pw):
        sh, an = [], []
        if cw and cw != "N/A":
            try:
                p = float(cw)
                sh.append(dict(type="line", x0=0, x1=1, xref="paper", y0=p, y1=p, line=dict(color="#FF6347", width=1, dash="dash")))
                an.append(dict(xref="paper", x=0.99, y=p, text=f"🔥 Call {p}", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color="#FF6347", size=11)))
            except: pass
        if pw and pw != "N/A":
            try:
                p = float(pw)
                sh.append(dict(type="line", x0=0, x1=1, xref="paper", y0=p, y1=p, line=dict(color="#3CB371", width=1, dash="dash")))
                an.append(dict(xref="paper", x=0.99, y=p, text=f"🛡️ Put {p}", showarrow=False, xanchor="right", yanchor="top", font=dict(color="#3CB371", size=11)))
            except: pass
        return sh, an

    shapes_common, annotations_common = get_wall_shapes_annotations(call_wall, put_wall)

    with tab1: # 周線
        try:
            df = stock.history(period="max", interval="1wk")
            if len(df) > 0:
                df['MA60'] = df['Close'].rolling(60).mean()
                if is_box_mode and vcp_weeks > 0 and len(df) >= vcp_weeks + 1:
                    last_n = df.iloc[-(vcp_weeks+1):-1]
                    if len(last_n) > 0:
                        box_shapes.append(dict(type="rect", x0=last_n.index[0], y0=last_n['Low'].min(), x1=last_n.index[-1], y1=last_n['High'].max(), line=dict(width=0), fillcolor="rgba(30, 144, 255, 0.25)"))
                fig = go.Figure([
                    go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='周K'),
                    go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=2))
                ])
                fig.update_layout(title=f"  {symbol} 周線", shapes=shapes_common + box_shapes, annotations=annotations_common, **layout_common)
                if len(df) > 150: fig.update_xaxes(range=[df.index[-150], df.index[-1]])
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"周線圖錯誤: {e}")

    with tab2: # 日線 (優化：使用整數索引解決斷層與拖曳)
        try:
            df = stock.history(period="5y")
            if len(df) > 0:
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA60'] = df['Close'].rolling(60).mean()
                
                # 使用整數索引重建 DataFrame，消除假日空隙
                df['d_str'] = df.index.strftime('%Y-%m-%d')
                df = df.reset_index(drop=True)
                
                fig = go.Figure([
                    go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='日K'),
                    go.Scatter(x=df.index, y=df['MA5'], mode='lines', name='MA5', line=dict(color='cyan', width=1), connectgaps=True),
                    go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='yellow', width=1), connectgaps=True),
                    go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=2), connectgaps=True)
                ])
                
                # 重新映射 X 軸
                tick_vals = np.arange(0, len(df), max(1, len(df)//8))
                tick_text = [df['d_str'].iloc[i] for i in tick_vals]
                
                fig.update_layout(title=f"  {symbol} 日線", shapes=shapes_common, annotations=annotations_common, **layout_common)
                fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=[max(0, len(df)-200), len(df)+5])
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"日線圖錯誤: {e}")

    with tab3: # 4H (已優化)
        try:
            df_1h = stock.history(period="1y", interval="1h")
            if len(df_1h) > 0:
                df = df_1h.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
                df['MA60'] = df['Close'].rolling(60).mean()
                df['d_str'] = df.index.strftime('%m-%d %H:%M')
                df = df.reset_index(drop=True)
                fig = go.Figure([
                    go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='4H K'),
                    go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=2), connectgaps=True)
                ])
                tick_vals = np.arange(0, len(df), max(1, len(df)//6))
                tick_text = [df['d_str'].iloc[i] for i in tick_vals]
                fig.update_layout(title=f"  {symbol} 4H", shapes=shapes_common, annotations=annotations_common, **layout_common)
                fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=[max(0, len(df)-160), len(df)+5])
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"4H 圖錯誤: {e}")

# --- 6. 核心運算 (強化版趨勢濾網 V3.0) ---
def get_ghost_metrics(symbol, vol_threshold, s, debug=False):
    def reject(reason): 
        return {"type": "error", "代號": symbol, "原因": reason} if debug else None

    try:
        stock = yf.Ticker(symbol)
        df_daily_2y = stock.history(period="2y", interval="1d")
        if df_daily_2y.empty: return reject("無法抓取資料 (Empty)")
        if len(df_daily_2y) < 250: return reject("資料不足 250 天")
        
        # 基礎計算
        curr_price = df_daily_2y['Close'].iloc[-1]
        log_ret = np.log(df_daily_2y['Close'] / df_daily_2y['Close'].shift(1))
        vol_30d = log_ret.rolling(30).std() * np.sqrt(252) * 100
        hv_rank_val = ((vol_30d.iloc[-1] - vol_30d.min()) / (vol_30d.max() - vol_30d.min())) * 100
        ma60_4h_val, dist_pct_val = 0, 0
        final_box_weeks = 0 
        ma5_cross_days_str = None
        ma5_cross_days_val = 999 
        status_note = ""
        sort_val = 0

        # --- A. 霸道模式 (箱型) ---
        if s['enable_box_breakout']:
            df_wk = df_daily_2y.resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            if len(df_wk) < 15: return reject("週線資料不足")
            avg_vol = df_wk['Volume'].tail(10).mean()
            if avg_vol < vol_threshold * 2: return reject(f"週均量不足 (需 > {int(vol_threshold*2)})")
            
            candidate_periods = [52, 40, 30, 20, 12] if s['enable_full_auto_vcp'] else [s['box_weeks']]
            found_vcp = False; box_str = ""; box_amp_str = ""
            current_week = df_wk.iloc[-1]
            
            for p in candidate_periods:
                if len(df_wk) < p + 2: continue
                box_data = df_wk.iloc[-(p+1):-1]
                box_high = box_data['High'].max(); box_low = box_data['Low'].min()
                if box_low == 0: continue
                if s['auto_flag_mode'] or s['enable_full_auto_vcp']:
                    mid = len(box_data)//2
                    old_r = box_data.iloc[:mid]['High'].max() - box_data.iloc[:mid]['Low'].min()
                    new_r = box_data.iloc[mid:]['High'].max() - box_data.iloc[mid:]['Low'].min()
                    if old_r == 0: continue
                    if new_r > old_r * 0.85: continue 
                    if current_week['Close'] < box_high * 0.90: continue 
                    if current_week['Close'] < box_high * 0.98: continue 
                    found_vcp = True; final_box_weeks = p; box_str = f"突破 {round(box_high, 2)}"; box_amp_str = f"VCP{p}W"; break
                else: 
                    amp = (box_high - box_low) / box_low * 100
                    if amp > s['box_tightness']: continue
                    if current_week['Close'] >= box_high * 0.99:
                        found_vcp = True; final_box_weeks = p; box_str = f"突破 {round(box_high, 2)}"; box_amp_str = f"{round(amp,1)}%"; break
            
            if not found_vcp: return reject("不符合 VCP/箱型型態")
            status_note = box_amp_str
            sort_val = 99999

        # --- B. 落水狗反彈模式 ---
        elif s['enable_reversal_mode']:
            df_daily_2y['MA5'] = df_daily_2y['Close'].rolling(5).mean()
            df_daily_2y['MA60'] = df_daily_2y['Close'].rolling(60).mean()
            
            curr = df_daily_2y.iloc[-1]
            prev_10 = df_daily_2y.iloc[-10]
            prev_20 = df_daily_2y.iloc[-20]
            prev_40 = df_daily_2y.iloc[-40]

            if not (curr['MA60'] < prev_10['MA60'] < prev_20['MA60'] < prev_40['MA60']): return reject("MA60 沒有呈現持續下滑")
            if not (curr['MA5'] > curr['MA60']): return reject("目前 MA5 尚未突破 MA60")
                
            days_since_cross = -1
            for i in range(1, 16):
                idx = -1 - i
                row = df_daily_2y.iloc[idx]
                if row['MA5'] <= row['MA60']: 
                    days_since_cross = i
                    break
            
            if days_since_cross == -1: return reject("未在最近 15 天內發現黃金交叉點")
            ma5_cross_days_val = days_since_cross 
            ma5_cross_days_str = f"已突破 {days_since_cross} 天" if days_since_cross > 0 else "剛突破"
            week_vol = log_ret.tail(5).std()*np.sqrt(5)*100 if len(log_ret)>=5 else 0
            box_str = f"±{round(curr_price*(week_vol/100),2)}"
            box_amp_str = round(week_vol, 2)

            try:
                df_1h = stock.history(period="1y", interval="1h")
                if len(df_1h) > 200:
                    df_4h = df_1h.resample('4h').agg({'Close':'last'}).dropna()
                    df_4h['MA60'] = df_4h['Close'].rolling(60).mean()
                    ma60_4h_val = df_4h['MA60'].iloc[-1]
                    dist_pct_val = ((df_4h['Close'].iloc[-1]-ma60_4h_val)/ma60_4h_val)*100
            except: pass

        # --- C. 趨勢特快車 (魔鬼濾網版 V3.0) ---
        elif s['enable_trend_mode']:
            df_daily_2y['MA5'] = df_daily_2y['Close'].rolling(5).mean()
            df_daily_2y['MA20'] = df_daily_2y['Close'].rolling(20).mean()
            df_daily_2y['MA60'] = df_daily_2y['Close'].rolling(60).mean()
            df_daily_2y['MA120'] = df_daily_2y['Close'].rolling(120).mean()
            
            c = df_daily_2y.iloc[-1]
            
            # 1. 嚴格多頭排列 (連續 3 天確認，防止單日假突破)
            for i in range(1, 4):
                h = df_daily_2y.iloc[-i]
                if not (h['Close'] > h['MA5'] > h['MA20'] > h['MA60'] > h['MA120']):
                    return reject("未維持至少3天多頭排列")

            # 2. 扇形發散 (乖離率門檻提高，濾除黏滯股)
            # KMI 這種股票通常 MA5 和 MA20 黏很緊，這裡要求 MA5 > MA20 * 1.01 (1%)
            if not (c['MA5'] > c['MA20'] * 1.01):
                return reject(f"MA5/MA20 發散不足 ({round((c['MA5']/c['MA20']-1)*100,1)}% < 1%)")
            
            # MA20 必須拉開 MA60 至少 2%
            if not (c['MA20'] > c['MA60'] * 1.02):
                return reject(f"MA20/MA60 發散不足 ({round((c['MA20']/c['MA60']-1)*100,1)}% < 2%)")

            # 3. 攻擊角度 (Slope) - 提高門檻到 0.002
            ma20_recent = df_daily_2y['MA20'].tail(10).values
            ma20_norm = ma20_recent / ma20_recent[0] 
            x = np.arange(len(ma20_norm))
            slope, _ = np.polyfit(x, ma20_norm, 1)
            
            if slope < 0.0020:
                return reject(f"MA20 攻擊角度太平緩 (Slope {round(slope*10000)} < 20)")

            # 4. RSI 強勢確認 (濾除轉弱股)
            delta = df_daily_2y['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            if rsi < 55: return reject(f"RSI 動能不足 ({round(rsi)} < 55)")
            if rsi > 85: return reject(f"RSI 過熱風險 ({round(rsi)} > 85)")
            
            status_note = f"🚀 仰角{round(slope*10000)}"
            sort_val = slope 

            # 補齊 4H
            try:
                df_1h = stock.history(period="1y", interval="1h")
                if len(df_1h) > 200:
                    df_4h = df_1h.resample('4h').agg({'Close':'last'}).dropna()
                    df_4h['MA60'] = df_4h['Close'].rolling(60).mean()
                    ma60_4h_val = df_4h['MA60'].iloc[-1]
                    dist_pct_val = ((df_4h['Close'].iloc[-1]-ma60_4h_val)/ma60_4h_val)*100
            except: pass

        # --- D. 幽靈模式 (標準) ---
        else:
            df_1h = stock.history(period="1y", interval="1h")
            if len(df_1h) < 240: return reject("1H 資料不足")
            df_daily = df_1h.resample('D').agg({'Volume':'sum','Close':'last'}).dropna()
            df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
            
            if s['check_daily_ma60_up'] and df_daily['MA60'].iloc[-1] <= df_daily['MA60'].iloc[-2]: return reject("日線 60MA 下彎")
            if df_daily['Volume'].rolling(20).mean().iloc[-1] < vol_threshold: return reject("成交量不足")
            if s['check_price_above_daily_ma60'] and df_daily['Close'].iloc[-1] < df_daily['MA60'].iloc[-1]: return reject("股價低於日線 60MA")
            if hv_rank_val > s['hv_threshold']: return reject(f"HV Rank {round(hv_rank_val)} 過高")
            
            if "週線點火" in s['ignition_mode'] or s['check_ma60_strong_trend']:
                df_wk = df_daily_2y.resample('W').agg({'Close':'last','High':'max'}).dropna()
                if s['check_ma60_strong_trend']:
                    ma60_wk = df_wk['Close'].rolling(60).mean()
                    if len(ma60_wk)>5 and not ma60_wk.tail(5).is_monotonic_increasing: return reject("週線 MA60 未向上")
                if "週線點火" in s['ignition_mode'] and len(df_wk)>=3:
                    curr = df_daily_2y['Close'].iloc[-1]
                    last_h = df_wk['High'].iloc[-2]
                    last_c = df_wk['Close'].iloc[-2]
                    prev_h = df_wk['High'].iloc[-3]
                    if not (curr > last_h or last_c > prev_h): return reject("週線未點火 (未過前高)")

            df_4h = df_1h.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            df_4h['MA60'] = df_4h['Close'].rolling(60).mean()
            ma60_4h_val = df_4h['MA60'].iloc[-1]
            dist_pct_val = ((df_4h['Close'].iloc[-1]-ma60_4h_val)/ma60_4h_val)*100
            
            if abs(dist_pct_val) > s['dist_threshold']: return reject(f"4H 乖離率 {round(dist_pct_val,2)}% 過大")
            if "4H 點火" in s['ignition_mode'] and len(df_4h)>=2:
                if df_4h['Close'].iloc[-1] <= df_4h['High'].iloc[-2]: return reject("4H 未點火")
            
            if s['enable_u_logic']:
                y = df_4h['MA60'].tail(s['u_sensitivity']).values; x = np.arange(len(y))
                try:
                    a, b, c = np.polyfit(x, y, 2)
                    if a <= 0: return reject("U型失敗 (開口向下)")
                    if a < s['min_curvature']: return reject("U型失敗 (彎曲度不足)")
                except: return reject("U型計算錯誤")
            
            week_vol = log_ret.tail(5).std()*np.sqrt(5)*100 if len(log_ret)>=5 else 0
            box_str = f"±{round(curr_price*(week_vol/100),2)}"
            box_amp_str = round(week_vol, 2)
            status_note = box_amp_str
            sort_val = -abs(dist_pct_val)

        # --- 期權運算 (累積加總) ---
        atm_oi = "N/A"; c_max_strike = "N/A"; p_max_strike = "N/A"
        call_oi_map = {}; put_oi_map = {}
        try:
            opts = stock.options
            if opts:
                chain = stock.option_chain(opts[0])
                idx = (chain.calls['strike'] - curr_price).abs().idxmin()
                strike_atm = chain.calls.loc[idx, 'strike']
                tot_atm_oi = chain.calls[chain.calls['strike']==strike_atm]['openInterest'].sum() + \
                             chain.puts[chain.puts['strike']==strike_atm]['openInterest'].sum()
                atm_oi = f"{int(tot_atm_oi):,}"
                
                if tot_atm_oi < 1000: return reject(f"期權流動性不足 OI={tot_atm_oi}")

                for d in opts[:6]:
                    try:
                        ch = stock.option_chain(d)
                        if not ch.calls.empty:
                            for _, row in ch.calls.iterrows():
                                k = row['strike']; v = row['openInterest']
                                call_oi_map[k] = call_oi_map.get(k, 0) + (v if v else 0)
                        if not ch.puts.empty:
                            for _, row in ch.puts.iterrows():
                                k = row['strike']; v = row['openInterest']
                                put_oi_map[k] = put_oi_map.get(k, 0) + (v if v else 0)
                    except: continue
                
                if call_oi_map: c_max_strike = max(call_oi_map, key=call_oi_map.get)
                if put_oi_map: p_max_strike = max(put_oi_map, key=put_oi_map.get)
        except: pass

        earnings = "未知"
        if stock.calendar and 'Earnings Date' in stock.calendar:
            earnings = stock.calendar['Earnings Date'][0].strftime('%m-%d')

        return {
            "type": "success",
            "代號": symbol, "HV Rank": round(hv_rank_val,1), 
            "狀態/波動": status_note, 
            "_sort_val": sort_val, 
            "MA5突破天數": ma5_cross_days_str, 
            "_ma5_days": ma5_cross_days_val, 
            "現價": round(curr_price,2), 
            "4H 60MA": round(ma60_4h_val,2) if ma60_4h_val!=0 else "N/A",
            "4H MA60 乖離率": f"{round(dist_pct_val,2)}%" if ma60_4h_val!=0 else "N/A",
            "價平OI": atm_oi, "全Call大量": c_max_strike, "全Put大量": p_max_strike,
            "產業": translate_industry(stock.info.get('industry','N/A')), "下次財報": earnings,
            "題材搜尋": f"https://www.google.com/search?q={symbol}+題材+風險",
            "_sort_score": 99999 if s['enable_box_breakout'] else -abs(dist_pct_val),
            "_vcp_weeks": final_box_weeks
        }
    except Exception as e:
        return reject(f"程式錯誤: {str(e)}")

# --- 7. 抓取代號 ---
@st.cache_data(ttl=3600)
def get_tickers_robust(choice):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        if "S&P" in choice:
            df = pd.read_html(StringIO(requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers).text))[0]
            return df[df.columns[0]].tolist()
        elif "NASDAQ" in choice:
            dfs = pd.read_html(StringIO(requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers).text))
            for d in dfs: 
                if 95 <= len(d) <= 105: return d[d.columns[0]].tolist()
        else:
            t1 = get_tickers_robust("S&P 500"); t2 = get_tickers_robust("NASDAQ 100")
            return list(set(t1 + t2))
    except: return ["AAPL","NVDA","TSLA","AMD","MSFT","GOOG","AMZN","META"]

# --- 8. 主程式 ---
if st.button("🚀 啟動 Turbo 掃描", type="primary"):
    st.session_state['scan_results'] = None
    status_text = "🔍 掃描中..."
    
    error_list = []

    with st.status(status_text, expanded=True) as status:
        tickers = get_tickers_robust(market_choice)[:scan_limit]
        status.write(f"✅ 已獲得 {len(tickers)} 檔代號，開始過濾...")
        results = []; count = 0; progress = st.progress(0)
        
        workers = 1 if debug_mode else max_workers
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_ticker = {executor.submit(get_ghost_metrics, t, st.session_state.min_vol_m*1000000, settings, debug_mode): t for t in tickers}
            
            for future in as_completed(future_to_ticker):
                data = future.result()
                count += 1
                progress.progress(count / len(tickers))
                
                if data and data.get("type") == "success":
                    data.pop("type")
                    results.append(data)
                elif data and data.get("type") == "error":
                    error_list.append({"代號": data["代號"], "原因": data["原因"]})

        if debug_mode and error_list:
            status.markdown("---")
            status.warning(f"📉 **篩選失敗清單 (共 {len(error_list)} 檔)**")
            err_df = pd.DataFrame(error_list)
            status.dataframe(err_df, height=300, use_container_width=True, hide_index=True)

        st.session_state['scan_results'] = results
        status.update(label=f"完成！共 {len(results)} 檔。", state="complete", expanded=False)

if 'scan_results' in st.session_state and st.session_state['scan_results']:
    df = pd.DataFrame(st.session_state['scan_results'])
    
    if settings.get('enable_reversal_mode'):
        if "_ma5_days" in df.columns: df = df.sort_values(by="_ma5_days", ascending=True)
    else:
        if "_sort_val" in df.columns: df = df.sort_values(by="_sort_val", ascending=False if settings.get('enable_trend_mode') else True)

    st.subheader("📋 策略篩選列表")
    
    df_display = df.copy()
    df_display["代號"] = df_display["代號"].apply(lambda x: f"https://finance.yahoo.com/quote/{x}/key-statistics")

    st.dataframe(df_display, column_config={
        "代號": st.column_config.LinkColumn("代號", display_text="https://finance\\.yahoo\\.com/quote/(.*?)/key-statistics"),
        "題材搜尋": st.column_config.LinkColumn("題材", display_text="🔍"),
        "_sort_val": None, "_sort_score": None, "_vcp_weeks": None, "_ma5_days": None 
    }, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🕯️ K 線檢視")
    
    options = df.apply(lambda x: f"{x['代號']} - {x['產業']}", axis=1).tolist()
    if options:
        sel = st.pills("👉 點擊標的", options, selection_mode="single")
        if sel:
            target = sel.split(" - ")[0]
            row = df[df['代號'] == target].iloc[0]
            plot_interactive_chart(target, row['全Call大量'], row['全Put大量'], row.get('_vcp_weeks', 0))
    else: st.write("查無標的")
