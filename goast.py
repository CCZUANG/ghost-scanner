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

# --- 2. 核心策略導引區 (確保縮排正確) ---
with st.expander("📖 點擊展開：幽靈策略動態蝴蝶演化步驟 (詳細準則)", expanded=False):
    col_step1, col_step2, col_step3 = st.columns(3)
    with col_step1:
        st.markdown("### 第一步：建立試探部位 (Rule 1)\n**🚀 啟動時機**：放量突破關鍵壓力。\n**動作**：買低 Call + 賣高 Call。")
    with col_step2:
        st.markdown("### 第二步：動能加碼 (Rule 2)\n**🚀 啟動時機**：浮盈且 IV 擴張。\n**動作**：加買更高一階 Call。")
    with col_step3:
        st.markdown("### 第三步：轉化蝴蝶 (退出方案)\n**🚀 啟動時機**：過熱訊號出現。\n**動作**：賣出中間價位 Call 鎖定成本。")
    st.info("💡 **核心注意事項**：Step 2 重點在於 IV 擴張。")

st.markdown("---")

# --- 3. 側邊欄 ---
st.sidebar.header("🎯 市場與數量")
market_choice = st.sidebar.radio("市場", ["S&P 500", "NASDAQ 100", "🔥 全火力"], index=2)
scan_limit = st.sidebar.slider("掃描數量", 50, 600, key='scan_limit')
debug_mode = st.sidebar.checkbox("🐞 啟動除錯模式 (顯示錯誤)", value=False)

settings = {}

st.sidebar.header("📦 箱型突破 (霸道模式)")
enable_box_breakout = st.sidebar.checkbox("✅ 啟動週線橫盤突破", value=False, key='box_mode_key', on_change=sync_logic_state)
settings['enable_box_breakout'] = enable_box_breakout

if enable_box_breakout:
    enable_full_auto_vcp = st.sidebar.checkbox("🤯 全自動 VCP 偵測", value=True)
    settings['enable_full_auto_vcp'] = enable_full_auto_vcp
    if not enable_full_auto_vcp:
        box_weeks = st.sidebar.slider("設定盤整週數 (N)", 4, 52, 20)
        settings['box_weeks'] = box_weeks
        auto_flag_mode = st.sidebar.checkbox("🤖 自動偵測旗型", value=True)
        settings['auto_flag_mode'] = auto_flag_mode
        settings['box_tightness'] = 100 if auto_flag_mode else st.sidebar.slider("寬度限制 (%)", 10, 50, 25)
    else:
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
    settings['min_curvature'] = st.sidebar.slider("最小彎曲度", 0
