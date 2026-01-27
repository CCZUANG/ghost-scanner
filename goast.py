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

# --- 2. 核心策略導引區 ---
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
            settings['box_weeks'] = 52 
            settings['auto_flag_mode'] = True
            settings['box_tightness'] = 100
else:
    settings['enable_full_auto_vcp'] = False
    settings['box_weeks'] = 52
    settings['auto_flag_mode'] = False
    settings['box_tightness'] = 25

# B. 幽靈模式
enable_u_logic = st.sidebar.checkbox("👻 啟動：U型/勺子 幽靈戰法", value=False, key='u_logic_key', on_change=handle_u_logic_toggle)
settings['enable_u_logic'] = enable_u_logic

if enable_u_logic:
    with st.sidebar.container(border=True):
        st.checkbox("🥄 嚴格勺子模式", value=True, key='spoon_strict_key', on_change=handle_spoon_toggle)
        settings['spoon_strict'] = st.session_state.spoon_strict_key
        settings['spoon_vertex_range'] = st.slider("底部位置 (%)", 0, 100, (50, 95), 5)
        
        st.markdown("---")
        settings['u_sensitivity'] = st.slider("U型敏感度 (Lookback)", 20, 240, key='u_sensitivity')
        settings['min_curvature'] = st.slider("最小彎曲度 (Curvature)", 0.0, 0.1, 0.003, format="%.3f")
else: 
    settings['spoon_strict'] = False
    settings['spoon_vertex_range'] = (50, 95)
    settings['u_sensitivity'] = 30
    settings['min_curvature'] = 0.003

st.sidebar.divider()

st.sidebar.subheader("🛡️ 3. 趨勢與濾網")

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    settings['check_daily_ma60_up'] = st.checkbox("日60MA向上", value=True)
    settings['check_price_above_daily_ma60'] = st.checkbox("股價 > 日MA", value=True)
with col_t2:
    settings['check_ma60_strong_trend'] = st.checkbox("週趨勢強勢", value=True)

ignition_mode = st.sidebar.radio("動能點火週期:", ["🚫 不啟用", "⚡ 4H 點火", "🚀 週線點火"], index=0, horizontal=True, key="ignition_mode_key", on_change=sync_logic_state)
settings['ignition_mode'] = ignition_mode

with st.sidebar.expander("⚙️ 進階參數 (波動率/成交量/效能)", expanded=False):
    st.caption("調整基礎過濾門檻")
    settings['hv_threshold'] = st.slider("HV Rank 上限", 10, 100, 30)
    min_vol_m = st.slider("最小日均量 (百萬股)", 1, 100, key='min_vol_m') 
    dist_threshold = st.slider("距離 4H MA60 容許範圍 (%)", 0.0, 50.0, step=0.5, key='dist_threshold')
    settings['dist_threshold'] = dist_threshold
    st.markdown("---")
    max_workers = st.slider("🚀 平行運算核心數", 1, 32, 16, help="數字越大掃描越快，但電腦負載越高")

# --- 4. 產業翻譯 ---
def translate_industry(eng):
    if not eng: return "未知"
    mp = {"technology":"科技","software":"軟體","financial":"金融","healthcare":"醫療","energy":"能源","industrials":"工業","real estate":"房產"}
    for k,v in mp.items():
        if k in eng.lower(): return v
    return eng

# --- 5. 繪圖函數 (4H 拖曳版 + 標籤上下分流) ---
def plot_interactive_chart(symbol, call_wall, put_wall, vcp_weeks=0, *args, **kwargs):
    stock = yf.Ticker(symbol)
    tab1, tab2, tab3 = st.tabs(["🗓️ 周線", "📅 日線", "⏱️ 4H"])
    
    # 手機 Layout: 圖例透明, 右側適中
    layout_common = dict(
        xaxis_rangeslider_visible=False, 
        height=500, 
        margin=dict(l=0, r=60, t=30, b=20), 
        legend=dict(orientation="h", y=0.99, x=0.01, bgcolor="rgba(0,0,0,0)"), 
        dragmode='pan'
    )
    
    box_shapes = []
    is_box_mode = st.session_state.get('box_mode_key', False)
    
    # 【優化】Call/Put 上下分流邏輯
    def get_wall_shapes_annotations(cw, pw):
        sh, an = [], []
        if cw and cw != "N/A":
            try:
                p = float(cw)
                sh.append(dict(type="line", x0=0, x1=1, xref="paper", y0=p, y1=p, line=dict(color="#FF6347", width=1, dash="dash")))
                # Call: yanchor="bottom" -> 文字在線的上方
                an.append(dict(xref="paper", x=0.99, y=p, text=f"🔥 Call {p}", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color="#FF6347", size=11)))
            except: pass
        if pw and pw != "N/A":
            try:
                p = float(pw)
                sh.append(dict(type="line", x0=0, x1=1, xref="paper", y0=p, y1=p, line=dict(color="#3CB371", width=1, dash="dash")))
                # Put: yanchor="top" -> 文字在線的下方
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
                        box_shapes.append(dict(
                            type="rect", x0=last_n.index[0], y0=last_n['Low'].min(), x1=last_n.index[-1], y1=last_n['High'].max(), 
                            line=dict(width=0), fillcolor="rgba(30, 144, 255, 0.25)"
                        ))
                fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='周K'),
                                 go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=2))])
                fig.update_layout(title=f"  {symbol} 周線", shapes=shapes_common + box_shapes, annotations=annotations_common, **layout_common)
                if len(df) > 150: fig.update_xaxes(range=[df.index[-150], df.index[-1]])
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"周線圖錯誤: {e}")

    with tab2: # 日線
        try:
            df = stock.history(period="5y")
            if len(df) > 0:
                df['MA60'] = df['Close'].rolling(60).mean()
                fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='日K'),
                                 go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=2))])
                fig.update_layout(title=f"  {symbol} 日線", shapes=shapes_common, annotations=annotations_common, **layout_common)
                if len(df) > 200: fig.update_xaxes(range=[df.index[-200], df.index[-1]])
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"日線圖錯誤: {e}")

    with tab3: # 4H (優化版：整數索引+拖曳)
        try:
            df_1h = stock.history(period="1y", interval="1h")
            if len(df_1h) > 0:
                df = df_1h.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
                df['MA60'] = df['Close'].rolling(60).mean()
                df['d_str'] = df.index.strftime('%m-%d %H:%M')
                
                df = df.reset_index(drop=True) # 使用整數索引以消除橫線並允許拖曳
                
                fig = go.Figure([
                    go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='4H K'),
                    go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=2))
                ])
                
                tick_vals = np.arange(0, len(df), max(1, len(df)//6))
                tick_text = [df['d_str'].iloc[i] for i in tick_vals]
                
                fig.update_layout(title=f"  {symbol} 4H", shapes=shapes_common, annotations=annotations_common, **layout_common)
                fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=[max(0, len(df)-160), len(df)+5])
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"4H 圖錯誤: {e}")

# --- 6. 核心運算 (修正版：期權 OI 加總) ---
def get_ghost_metrics(symbol, vol_threshold, s, debug=False):
    def reject(reason): 
        return {"type": "error", "代號": symbol, "原因": reason} if debug else None

    try:
        stock = yf.Ticker(symbol)
        df_daily_2y = stock.history(period="2y", interval="1d")
        if df_daily_2y.empty: return reject("無法抓取資料 (Empty)")
        if len(df_daily_2y) < 250: return reject("資料不足 250 天")
        
        # 2. 基礎數據
        curr_price = df_daily_2y['Close'].iloc[-1]
        log_ret = np.log(df_daily_2y['Close'] / df_daily_2y['Close'].shift(1))
        vol_30d = log_ret.rolling(30).std() * np.sqrt(252) * 100
        hv_rank_val = ((vol_30d.iloc[-1] - vol_30d.min()) / (vol_30d.max() - vol_30d.min())) * 100
        ma60_4h_val, dist_pct_val = 0, 0
        final_box_weeks = 0 

        # --- A. 霸道模式 ---
        if s['enable_box_breakout']:
            df_wk = df_daily_2y.resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            if len(df_wk) < 15: return reject("週線資料不足")
            
            avg_vol = df_wk['Volume'].tail(10).mean()
            if avg_vol < vol_threshold * 2: return reject(f"週均量不足 (需 > {int(vol_threshold*2)})")
            
            candidate_periods = [52, 40, 30, 20, 12] if s['enable_full_auto_vcp'] else [s['box_weeks']]
            found_vcp = False
            box_str = ""; box_amp_str = ""
            current_week = df_wk.iloc[-1]
            
            for p in candidate_periods:
                if len(df_wk) < p + 2: continue
                box_data = df_wk.iloc[-(p+1):-1]
                box_high = box_data['High'].max()
                box_low = box_data['Low'].min()
                if box_low == 0: continue
                
                if s['auto_flag_mode'] or s['enable_full_auto_vcp']:
                    mid = len(box_data)//2
                    old_r = box_data.iloc[:mid]['High'].max() - box_data.iloc[:mid]['Low'].min()
                    new_r = box_data.iloc[mid:]['High'].max() - box_data.iloc[mid:]['Low'].min()
                    
                    if old_r == 0: continue
                    if new_r > old_r * 0.85: continue 
                    if current_week['Close'] < box_high * 0.90: continue 
                    if current_week['Close'] < box_high * 0.98: continue 
                    
                    found_vcp = True; final_box_weeks = p
                    box_str = f"突破 {round(box_high, 2)}"; box_amp_str = f"VCP{p}W"
                    break
                else: 
                    amp = (box_high - box_low) / box_low * 100
                    if amp > s['box_tightness']: continue
                    if current_week['Close'] >= box_high * 0.99:
                        found_vcp = True; final_box_weeks = p
                        box_str = f"突破 {round(box_high, 2)}"; box_amp_str = f"{round(amp,1)}%"
                        break
            
            if not found_vcp: return reject("不符合 VCP/箱型型態")
            
            try:
                df_1h = stock.history(period="1y", interval="1h")
                if len(df_1h) > 200:
                    df_4h = df_1h.resample('4h').agg({'Close':'last'}).dropna()
                    df_4h['MA60'] = df_4h['Close'].rolling(60).mean()
                    ma60_4h_val = df_4h['MA60'].iloc[-1]
                    dist_pct_val = ((df_4h['Close'].iloc[-1]-ma60_4h_val)/ma60_4h_val)*100
            except: pass

        # --- B. 幽靈模式 ---
        else:
            df_1h = stock.history(period="1y", interval="1h")
            if len(df_1h) < 240: return reject("1H 資料不足")
            df_daily = df_1h.resample('D').agg({'Volume':'sum','Close':'last'}).dropna()
            df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
            
            if s['check_daily_ma60_up'] and df_daily['MA60'].iloc[-1] <= df_daily['MA60'].iloc[-2]: 
                return reject("日線 60MA 下彎")
            if df_daily['Volume'].rolling(20).mean().iloc[-1] < vol_threshold: 
                return reject("成交量不足")
            if s['check_price_above_daily_ma60'] and df_daily['Close'].iloc[-1] < df_daily['MA60'].iloc[-1]: 
                return reject("股價低於日線 60MA")
            if hv_rank_val > s['hv_threshold']: 
                return reject(f"HV Rank {round(hv_rank_val)} 過高")
            
            if "週線點火" in s['ignition_mode'] or s['check_ma60_strong_trend']:
                df_wk = df_daily_2y.resample('W').agg({'Close':'last','High':'max'}).dropna()
                if s['check_ma60_strong_trend']:
                    ma60_wk = df_wk['Close'].rolling(60).mean()
                    if len(ma60_wk)>5 and not ma60_wk.tail(5).is_monotonic_increasing: 
                        return reject("週線 MA60 未向上")
                if "週線點火" in s['ignition_mode'] and len(df_wk)>=3:
                    curr = df_daily_2y['Close'].iloc[-1]
                    last_h = df_wk['High'].iloc[-2]
                    last_c = df_wk['Close'].iloc[-2]
                    prev_h = df_wk['High'].iloc[-3]
                    if not (curr > last_h or last_c > prev_h): 
                        return reject("週線未點火 (未過前高)")

            df_4h = df_1h.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            df_4h['MA60'] = df_4h['Close'].rolling(60).mean()
            ma60_4h_val = df_4h['MA60'].iloc[-1]
            dist_pct_val = ((df_4h['Close'].iloc[-1]-ma60_4h_val)/ma60_4h_val)*100
            
            if abs(dist_pct_val) > s['dist_threshold']: 
                return reject(f"4H 乖離率 {round(dist_pct_val,2)}% 過大")
            
            if "4H 點火" in s['ignition_mode'] and len(df_4h)>=2:
                if df_4h['Close'].iloc[-1] <= df_4h['High'].iloc[-2]: 
                    return reject("4H 未點火")
            
            if s['enable_u_logic']:
                y = df_4h['MA60'].tail(s['u_sensitivity']).values; x = np.arange(len(y))
                try:
                    a, b, c = np.polyfit(x, y, 2)
                    if a <= 0: return reject("U型失敗 (開口向下)")
                    if a < s['min_curvature']: return reject("U型失敗 (彎曲度不足)")
                except: return reject("U型計算錯誤")
                
            week_vol = log_ret.tail(5).std()*np.sqrt(5)*100 if len(log_ret)>=5 else 0
            box_str = f"±{round(df_daily_2y['Close'].iloc[-1]*(week_vol/100),2)}"
            box_amp_str = round(week_vol, 2)

        # --- 期權運算 (修正：累積加總) ---
        atm_oi = "N/A"; c_max_strike = "N/A"; p_max_strike = "N/A"
        
        # 使用字典來累積不同日期的 OI
        call_oi_map = {} 
        put_oi_map = {}
        
        try:
            opts = stock.options
            if opts:
                # 1. 計算價平 OI (維持只看最近月，因為這是流動性指標)
                chain = stock.option_chain(opts[0])
                idx = (chain.calls['strike'] - curr_price).abs().idxmin()
                strike_atm = chain.calls.loc[idx, 'strike']
                tot_atm_oi = chain.calls[chain.calls['strike']==strike_atm]['openInterest'].sum() + \
                             chain.puts[chain.puts['strike']==strike_atm]['openInterest'].sum()
                atm_oi = f"{int(tot_atm_oi):,}"
                
                if tot_atm_oi < 1000: # 稍微放寬這裡的檢查
                     return reject(f"期權流動性不足 OI={tot_atm_oi}")

                # 2. 累積全期限 OI (掃描前 6 個月/週)
                for d in opts[:6]:
                    try:
                        ch = stock.option_chain(d)
                        # 累積 Call
                        if not ch.calls.empty:
                            for _, row in ch.calls.iterrows():
                                k = row['strike']
                                v = row['openInterest']
                                call_oi_map[k] = call_oi_map.get(k, 0) + (v if v else 0)
                        # 累積 Put
                        if not ch.puts.empty:
                            for _, row in ch.puts.iterrows():
                                k = row['strike']
                                v = row['openInterest']
                                put_oi_map[k] = put_oi_map.get(k, 0) + (v if v else 0)
                    except: continue
                
                # 3. 找出累積後的最大量價位
                if call_oi_map:
                    c_max_strike = max(call_oi_map, key=call_oi_map.get)
                if put_oi_map:
                    p_max_strike = max(put_oi_map, key=put_oi_map.get)
                    
        except: pass

        earnings = "未知"
        if stock.calendar and 'Earnings Date' in stock.calendar:
            earnings = stock.calendar['Earnings Date'][0].strftime('%m-%d')

        return {
            "type": "success",
            "代號": symbol, "HV Rank": round(hv_rank_val,1), "週波動%": box_amp_str, "預期變動$": box_str,
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
    df = pd.DataFrame(st.session_state['scan_results']).sort_values(by="HV Rank")
    st.subheader("📋 策略篩選列表")
    
    df_display = df.copy()
    df_display["代號"] = df_display["代號"].apply(lambda x: f"https://finance.yahoo.com/quote/{x}/key-statistics")

    st.dataframe(df_display, column_config={
        "代號": st.column_config.LinkColumn("代號", display_text="https://finance\\.yahoo\\.com/quote/(.*?)/key-statistics"),
        "題材搜尋": st.column_config.LinkColumn("題材", display_text="🔍"),
        "_sort_score": None, "_vcp_weeks": None
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

