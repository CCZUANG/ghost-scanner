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

# --- 2. 核心策略導引區 (詳細版) ---
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
scan_limit = st.sidebar.slider("掃描數量", 50, 600, key='scan_limit')

# 【新增】UI 除錯模式開關
debug_mode = st.sidebar.checkbox("🐞 啟動詳細除錯模式 (顯示失敗原因)", value=False, help="開啟後會顯示每一檔股票為什麼被篩選掉，速度會變慢")

settings = {}

st.sidebar.header("📦 箱型突破 (霸道模式)")
enable_box_breakout = st.sidebar.checkbox("✅ 啟動週線橫盤突破 (忽略其他條件)", value=False, key='box_mode_key', on_change=sync_logic_state)
settings['enable_box_breakout'] = enable_box_breakout

if enable_box_breakout:
    enable_full_auto_vcp = st.sidebar.checkbox("🤯 全自動 VCP 偵測 (免設定週數)", value=True)
    settings['enable_full_auto_vcp'] = enable_full_auto_vcp
    
    if not enable_full_auto_vcp:
        box_weeks = st.sidebar.slider("設定盤整週數 (N)", 4, 52, 20)
        settings['box_weeks'] = box_weeks
        auto_flag_mode = st.sidebar.checkbox("🤖 自動偵測旗型收斂", value=True)
        settings['auto_flag_mode'] = auto_flag_mode
        settings['box_tightness'] = 100 if auto_flag_mode else st.sidebar.slider("盤整區間寬度限制 (%)", 10, 50, 25)
    else:
        st.sidebar.caption("👉 系統將自動尋找最佳的收斂突破週期")
        settings['box_weeks'] = 52 
        settings['auto_flag_mode'] = True
        settings['box_tightness'] = 100
else:
    settings['enable_full_auto_vcp'] = False
    settings['box_weeks'] = 52
    settings['auto_flag_mode'] = False
    settings['box_tightness'] = 25

st.sidebar.divider()
st.sidebar.header("📈 幽靈戰法連動")
enable_u_logic = st.sidebar.checkbox("✅ 啟動 4小時 U型戰法", value=False, key='u_logic_key', on_change=handle_u_logic_toggle)
settings['enable_u_logic'] = enable_u_logic

if enable_u_logic:
    st.sidebar.checkbox("🥄 嚴格勺子模式", value=True, key='spoon_strict_key', on_change=handle_spoon_toggle)
    settings['spoon_strict'] = st.session_state.spoon_strict_key
    settings['spoon_vertex_range'] = st.sidebar.slider("🥄 勺子底部位置 (%)", 0, 100, (50, 95), 5)
else: 
    settings['spoon_strict'] = False
    settings['spoon_vertex_range'] = (50, 95)

st.sidebar.header("🛡️ 趨勢與點火")
settings['check_daily_ma60_up'] = st.sidebar.checkbox("✅ 日線 60MA 向上", value=True)
settings['check_ma60_strong_trend'] = st.sidebar.checkbox("✅ 週線 MA60 強勢趨勢", value=True)
settings['check_price_above_daily_ma60'] = st.sidebar.checkbox("✅ 股價 > 日線 60MA", value=True)
ignition_mode = st.sidebar.radio("動能點火週期:", ["🚫 不啟用", "⚡ 4H 點火", "🚀 週線點火"], index=0, key="ignition_mode_key", on_change=sync_logic_state)
settings['ignition_mode'] = ignition_mode

st.sidebar.header("⚙️ 基礎篩選")
settings['hv_threshold'] = st.sidebar.slider("HV Rank 門檻", 10, 100, 30)
min_vol_m = st.sidebar.slider("最小日均量 (百萬股)", 1, 100, key='min_vol_m') 
dist_threshold = st.sidebar.slider("距離 4H MA60 範圍 (%)", 0.0, 50.0, key='dist_threshold', step=0.5)
settings['dist_threshold'] = dist_threshold

if enable_u_logic:
    settings['u_sensitivity'] = st.sidebar.slider("U型敏感度", 20, 240, key='u_sensitivity')
    settings['min_curvature'] = st.sidebar.slider("最小彎曲度", 0.0, 0.1, 0.003, format="%.3f")
else: 
    settings['u_sensitivity'] = 30
    settings['min_curvature'] = 0.003
max_workers = st.sidebar.slider("🚀 平行核心數", 1, 32, 16)

# --- 4. 產業翻譯 ---
def translate_industry(eng):
    if not eng: return "未知"
    mp = {"technology":"科技","software":"軟體","financial":"金融","healthcare":"醫療","energy":"能源","industrials":"工業","real estate":"房產"}
    for k,v in mp.items():
        if k in eng.lower(): return v
    return eng

# --- 5. 繪圖函數 (已修正：手機優化 + 修正 df['close'] 小寫錯誤) ---
def plot_interactive_chart(symbol, call_wall, put_wall, vcp_weeks=0):
    stock = yf.Ticker(symbol)
    tab1, tab2, tab3 = st.tabs(["🗓️ 周線", "📅 日線", "⏱️ 4H"])
    
    # 手機優化 Layout
    layout = dict(
        xaxis_rangeslider_visible=False, 
        height=600, 
        margin=dict(l=0, r=130, t=30, b=30), # 根據您的需求維持 r=130
        legend=dict(orientation="h", y=-0.1, x=0.5), 
        dragmode=False
    )
    
    box_shapes = []
    is_box_mode = st.session_state.get('box_mode_key', False)
    
    def get_wall_shapes_annotations(cw, pw):
        sh, an = [], []
        if cw and cw != "N/A":
            try:
                p = float(cw)
                sh.append(dict(type="line", x0=0, x1=1, xref="paper", y0=p, y1=p, line=dict(color="#FF6347", width=1, dash="dash")))
                an.append(dict(xref="paper", x=1.01, y=p, text=f"🔥 Call {p}", showarrow=False, xanchor="left", yanchor="bottom", yshift=10, font=dict(color="#FF6347", size=12)))
            except: pass
        if pw and pw != "N/A":
            try:
                p = float(pw)
                sh.append(dict(type="line", x0=0, x1=1, xref="paper", y0=p, y1=p, line=dict(color="#3CB371", width=1, dash="dash")))
                an.append(dict(xref="paper", x=1.01, y=p, text=f"🛡️ Put {p}", showarrow=False, xanchor="left", yanchor="top", yshift=-10, font=dict(color="#3CB371", size=12)))
            except: pass
        return sh, an

    shapes_common, annotations_common = get_wall_shapes_annotations(call_wall, put_wall)

    with tab1: # 周線
        try:
            df = stock.history(period="max", interval="1wk")
            if len(df) > 0:
                # 【修正重點】這裡必須用 Capital 'Close'，之前您截圖中是 'close' 會報錯
                df['MA60'] = df['Close'].rolling(60).mean()
                
                # VCP 區塊
                if is_box_mode and vcp_weeks > 0 and len(df) >= vcp_weeks + 1:
                    last_n = df.iloc[-(vcp_weeks+1):-1]
                    if len(last_n) > 0:
                        box_shapes.append(dict(
                            type="rect", 
                            x0=last_n.index[0], 
                            y0=last_n['Low'].min(), 
                            x1=last_n.index[-1], 
                            y1=last_n['High'].max(), 
                            line=dict(width=0), 
                            fillcolor="rgba(30, 144, 255, 0.25)"
                        ))

                fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='周K'),
                                 go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=2))])
                
                all_shapes = shapes_common + box_shapes
                fig.update_layout(title=f"{symbol} 周線", shapes=all_shapes, annotations=annotations_common, **layout)
                if len(df) > 150: fig.update_xaxes(range=[df.index[-150], df.index[-1]])
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"周線圖錯誤: {e}")
