import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import os
import datetime
from dateutil.relativedelta import relativedelta

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="TSMOM 量化操盤室 (字型修復版)", layout="wide")


# ==========================================
# 🔤 字型修復專區 (雲端亂碼救星)
# ==========================================
def install_chinese_font():
    # 定義字型檔案名稱 (思源黑體)
    font_path = "NotoSansTC-Regular.ttf"

    # 如果檔案不存在，就從 Google 下載
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC-Regular.ttf"
        try:
            # 顯示下載進度以免使用者以為當機
            # with st.spinner("☁️ 正在為雲端環境下載中文字型，請稍候..."):
            response = requests.get(url)
            with open(font_path, "wb") as f:
                f.write(response.content)
        except:
            pass  # 下載失敗則忽略，使用預設

    # 將下載的字型加入 Matplotlib
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        # 設定 Matplotlib 使用這個字型
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    else:
        # 如果真的下載失敗，回退到系統預設 (雖然可能還是亂碼，但至少不報錯)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']

    # 設定負號正常顯示
    plt.rcParams['axes.unicode_minus'] = False


# 執行字型設定
install_chinese_font()
# ==========================================

# --- 2. 側邊欄設定 ---
st.sidebar.header("🎛️ 參數設定中心")

# 定義常用代碼清單
TICKER_LIST = {
    "📝 自行輸入代碼 (Custom)": "CUSTOM",
    "--- 🇺🇸 美股熱門 ---": "sep1",
    "NVDA (輝達)": "NVDA",
    "TSLA (特斯拉)": "TSLA",
    "AAPL (蘋果)": "AAPL",
    "MSFT (微軟)": "MSFT",
    "AMZN (亞馬遜)": "AMZN",
    "GOOGL (谷歌)": "GOOGL",
    "AMD (超微)": "AMD",
    "PLTR (Palantir)": "PLTR",
    "MSTR (微策略)": "MSTR",
    "--- 🇺🇸 美股指數 ---": "sep2",
    "S&P 500 (標普)": "^GSPC",
    "Nasdaq 100 (納指)": "^NDX",
    "Dow Jones (道瓊)": "^DJI",
    "PHLX Semi (費半)": "^SOX",
    "--- 🇹🇼 台股熱門 ---": "sep3",
    "2330.TW (台積電)": "2330.TW",
    "2317.TW (鴻海)": "2317.TW",
    "2454.TW (聯發科)": "2454.TW",
    "2382.TW (廣達)": "2382.TW",
    "2603.TW (長榮)": "2603.TW",
    "--- 🇹🇼 台股指數 ---": "sep4",
    "Taiwan Weighted (加權指數)": "^TWII",
    "TPEx (櫃買指數)": "^TWO",
}

selected_label = st.sidebar.selectbox("🎯 快速選擇標的", options=list(TICKER_LIST.keys()))
ticker_val = TICKER_LIST[selected_label]

if ticker_val == "CUSTOM":
    ticker = st.sidebar.text_input("請輸入股票代碼", value="NVDA", help="例如: TSM, 2330.TW, ^TWII")
elif "sep" in ticker_val:
    st.sidebar.warning("⚠️ 請選擇有效的股票，不要選分隔線")
    ticker = "NVDA"
else:
    st.sidebar.text_input("目前選定代碼", value=ticker_val, disabled=True)
    ticker = ticker_val

years = st.sidebar.slider("回測年數", 1, 10, 5)
run_btn = st.sidebar.button("🚀 啟動實驗分析", type="primary")

st.sidebar.markdown("---")
st.sidebar.info("""
**圖表線條說明：**
1. 🔴 **紅線 (穩健)**：鄰居測試 + 控倉 (安全帶)
2. 🔵 **藍線 (狂暴)**：鄰居測試 + **全倉 (油門到底)**
3. 🟠 **橘線 (貪婪)**：無鄰居 + 控倉
4. 🟣 **紫線 (極致)**：無鄰居 + 全倉
5. ⚪ **灰線**：買進持有
""")


# --- 3. 核心函數 ---
@st.cache_data
def get_data(symbol, lookback_years):
    end = datetime.date.today()
    start = end - relativedelta(years=lookback_years)
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                price = df.xs('Adj Close', axis=1, level=0)
            except:
                price = df.xs('Close', axis=1, level=0)
        else:
            price = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        price = price.dropna(axis=1, how='all').ffill()
        if isinstance(price, pd.DataFrame): price = price.iloc[:, 0]
        return price
    except Exception:
        return None


def analyze_strategy(price):
    r = np.log(price.pct_change() + 1)

    # 1. 掃描所有均線績效
    performance_map = {}
    windows = range(10, 255, 1)

    for w in windows:
        sig = r.rolling(w).sum()
        pos = np.sign(sig.shift(1)).fillna(0)
        perf = (pos * r).sum()
        performance_map[w] = perf

    # --- 選參數 ---
    best_win_peak = max(performance_map, key=performance_map.get)

    best_win_robust = 60
    best_robust_score = -np.inf
    for w in windows:
        neighbors = [n for n in range(w - 2, w + 3) if n in performance_map]
        if not neighbors: continue
        avg_score = np.mean([performance_map[n] for n in neighbors])
        if avg_score > best_robust_score:
            best_robust_score = avg_score
            best_win_robust = w

    # --- 計算控倉係數 ---
    vol_window = 60
    ann_vol = r.rolling(vol_window).std() * (252 ** 0.5)
    target_vol = 0.40
    vol_scale = (target_vol / ann_vol).replace([np.inf, -np.inf], 0).fillna(0).clip(upper=1.0)

    # --- 產生訊號 ---
    mom_robust = r.rolling(best_win_robust).sum()
    sig_robust = np.sign(mom_robust.shift(1)).fillna(0)

    mom_peak = r.rolling(best_win_peak).sum()
    sig_peak = np.sign(mom_peak.shift(1)).fillna(0)

    # 台股防呆邏輯
    ticker_name = str(price.name).upper() if hasattr(price, 'name') else ""
    if ".TW" in ticker_name:
        sig_robust = pd.Series(np.where(sig_robust < 0, 0, sig_robust), index=r.index)
        sig_peak = pd.Series(np.where(sig_peak < 0, 0, sig_peak), index=r.index)

    # --- 產生曲線 ---
    cum_safe_robust = np.exp((sig_robust * vol_scale.shift(1).fillna(0) * r).cumsum())

    pos_turbo = sig_robust
    cum_turbo_robust = np.exp((pos_turbo * r).cumsum())

    cum_safe_peak = np.exp((sig_peak * vol_scale.shift(1).fillna(0) * r).cumsum())
    cum_turbo_peak = np.exp((sig_peak * r).cumsum())
    cum_hold = np.exp(r.cumsum())

    return {
        "safe_robust": cum_safe_robust,
        "turbo_robust": cum_turbo_robust,
        "safe_peak": cum_safe_peak,
        "turbo_peak": cum_turbo_peak,
        "hold": cum_hold,
        "win_robust": best_win_robust,
        "win_peak": best_win_peak,
        "curr_vol": ann_vol.iloc[-1],
        "curr_pos": pos_turbo.iloc[-1]
    }


# --- 4. 主畫面呈現 ---
st.title("🧪 TSMOM 量化操盤室：旗艦選單版")

if run_btn:
    with st.spinner(f'正在進行高精度掃描...'):
        price_data = get_data(ticker, years)

        if price_data is None or price_data.empty:
            st.error(f"❌ 找不到代碼：{ticker}，請確認是否輸入正確。")
        else:
            last_date = price_data.index[-1].strftime('%Y-%m-%d')
            st.success(f"📅 資料更新至：{last_date} (最新收盤價) | 代碼: {ticker}")

            res = analyze_strategy(price_data)

            # --- A. 顯示關鍵指標 ---
            c1, c2, c3, c4 = st.columns(4)
            ret_safe = (res["safe_robust"].iloc[-1] - 1) * 100
            ret_turbo = (res["turbo_robust"].iloc[-1] - 1) * 100

            c1.metric("🔴 穩健策略 (紅)", f"{ret_safe:.1f}%")
            c2.metric("🔵 狂暴策略 (藍)", f"{ret_turbo:.1f}%", delta=f"{ret_turbo - ret_safe:.1f}% (差距)")
            c3.metric("⚙️ 最佳均線", f"MA{res['win_robust']}")
            c4.metric("🌊 當前波動", f"{res['curr_vol'] * 100:.1f}%")

            # --- B. AI 操作建議卡片 ---
            st.subheader("💡 AI 操作建議 (基於藍線-狂暴策略)")

            curr_pos = res["curr_pos"]

            if curr_pos > 0:
                msg = f"🔵 **全力買進 (Turbo Long)**"
                bg_color = "rgba(50, 50, 255, 0.2)"
                pos_text = "100% (滿倉)"
            elif curr_pos < 0:
                msg = f"🟣 **全力放空 (Turbo Short)**"
                bg_color = "rgba(200, 50, 200, 0.2)"
                pos_text = "100% (滿倉)"
            else:
                msg = "⚫ **空手觀望 (Neutral)**"
                bg_color = "rgba(128, 128, 128, 0.2)"
                pos_text = "0%"

            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: {bg_color}; font-size: 20px; font-weight: bold; margin-bottom: 20px;">
                {msg} <br> <span style="font-size: 16px;">(建議倉位: {pos_text})</span>
            </div>
            """, unsafe_allow_html=True)

            # --- C. 繪圖區 ---
            st.subheader(f"📈 五線大亂鬥：{ticker}")
            fig, ax = plt.subplots(figsize=(12, 7))

            ax.plot(res["turbo_robust"], color='blue', linewidth=2.5, label='★ 藍線：穩健+狂暴 (主角)')
            ax.plot(res["safe_robust"], color='red', linewidth=1.5, alpha=0.8, label='紅線：穩健+控倉')
            ax.plot(res["safe_peak"], color='orange', linestyle=':', linewidth=1.5, label='橘線：貪婪+控倉')
            ax.plot(res["turbo_peak"], color='purple', linestyle=':', linewidth=1.5, label='紫線：貪婪+狂暴')
            ax.plot(res["hold"], color='gray', linestyle='--', alpha=0.4, label='買進持有')

            ax.set_title(f"策略對決：MA{res['win_robust']} (Current Best)", fontsize=14)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            st.warning("""
            **⚠️ 狂暴模式注意：**
            * 建議無視波動率，直接給予 **100% 滿倉** 的訊號。
            * 在牛市 (如 NVDA) 會賺非常多，但在盤整或空頭可能會面臨較大的回檔。
            """)
else:
    st.info("👈 請在左側選單選擇股票，或輸入代碼開始分析")