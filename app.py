import streamlit as st

# ==========================================
# 🩹 [修復補丁] 消除黃色警告框專區
# ==========================================
# 這段程式碼會把舊版指令「偷換」成新版，讓警告閉嘴
try:
    # 如果 Streamlit 版本較新，強行覆蓋舊函數，避免跳出 DeprecationWarning
    st.experimental_get_query_params = lambda: st.query_params
except AttributeError:
    pass
# ==========================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request
import os
import datetime
from dateutil.relativedelta import relativedelta
import streamlit_analytics

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="TSMOM 量化操盤室 (無痕戰情版)", layout="wide")

# ==========================================
# 📊 流量分析啟動區 (🔒 加密版)
# ==========================================
# 密碼設為 tsmom888 (您可自行修改)
with streamlit_analytics.track(unsafe_password="tsmom888"):
    # ==========================================
    # 🔤 字型修復專區 (Adobe 思源黑體)
    # ==========================================
    def install_chinese_font():
        font_path = "SourceHanSansTC-Regular.otf"
        url_primary = "https://raw.githubusercontent.com/adobe-fonts/source-han-sans/release/OTF/TraditionalChinese/SourceHanSansTC-Regular.otf"

        if not os.path.exists(font_path):
            try:
                urllib.request.urlretrieve(url_primary, font_path)
            except Exception as e:
                print(f"字型下載失敗: {e}")

        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
        else:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False


    install_chinese_font()
    # ==========================================

    # --- 2. 側邊欄設定 ---
    st.sidebar.header("🎛️ 參數設定中心")

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
        "SIVR (白銀ETF)": "SIVR",
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
        ticker = st.sidebar.text_input(
            "請輸入股票代碼 (Yahoo Finance 代碼)",
            value="NVDA",
            help="⚠️ 請注意：必須使用 Yahoo Finance 格式。\n例如：台股請加 .TW (2330.TW), 標普指數 (^GSPC)"
        )
    elif "sep" in ticker_val:
        st.sidebar.warning("⚠️ 請選擇有效的股票，不要選分隔線")
        ticker = "NVDA"
    else:
        st.sidebar.text_input("目前選定代碼", value=ticker_val, disabled=True)
        ticker = ticker_val

    years = st.sidebar.slider("回測年數", 1, 20, 20)
    run_btn = st.sidebar.button("🚀 啟動實驗分析", type="primary")

    st.sidebar.markdown("---")
    st.sidebar.info("請點擊主畫面中的「📖 使用說明書」來了解如何解讀圖表。")


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

        # 掃描績效
        performance_map = {}
        windows = range(10, 255, 1)
        for w in windows:
            sig = r.rolling(w).sum()
            pos = np.sign(sig.shift(1)).fillna(0)
            perf = (pos * r).sum()
            performance_map[w] = perf

        # 選參數
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

        # 計算波動率與控倉
        vol_window = 60
        ann_vol = r.rolling(vol_window).std() * (252 ** 0.5)
        target_vol = 0.40
        vol_scale = (target_vol / ann_vol).replace([np.inf, -np.inf], 0).fillna(0).clip(upper=1.0)

        # 產生訊號
        mom_robust = r.rolling(best_win_robust).sum()
        sig_robust = np.sign(mom_robust.shift(1)).fillna(0)
        mom_peak = r.rolling(best_win_peak).sum()
        sig_peak = np.sign(mom_peak.shift(1)).fillna(0)

        # 台股防呆
        ticker_name = str(price.name).upper() if hasattr(price, 'name') else ""
        if ".TW" in ticker_name:
            sig_robust = pd.Series(np.where(sig_robust < 0, 0, sig_robust), index=r.index)
            sig_peak = pd.Series(np.where(sig_peak < 0, 0, sig_peak), index=r.index)

        # 產生曲線
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
    st.title("🧪 TSMOM 量化操盤室")

    with st.expander("📖 新手必讀：TSMOM 原理、圖表解讀與操作攻略 (點擊展開)", expanded=False):
        st.markdown("""
        ### 1️⃣ TSMOM 原理與圖表解讀
        **TSMOM (時間序列動能)** 的邏輯就像牛頓慣性定律：**「趨勢形成後，傾向於持續。」**

        | 顏色 | 名稱 | 策略邏輯 | 風險屬性 | 誰適合用？ |
        | :--- | :--- | :--- | :--- | :--- |
        | 🔴 **紅線** | **穩健+控倉** | **找鄰居 (Robust)** + **波動率控倉**。 | **低 (防禦型)** | **穩健投資人** |
        | 🔵 **藍線** | **穩健+狂暴** | **找鄰居 (Robust)** + **全倉 (100%)**。 | **高 (攻擊型)** | **趨勢交易者** |
        | ⚪ **灰線** | **買進持有** | **Buy & Hold** (傻傻抱著)。 | **市場風險** | **存股族** |

        *(註：🟠 橘線與 🟣 紫線為理論最佳值，僅供對照參考)*

        ---

        ### 2️⃣ 關鍵：該回測多少年？ (不同資產建議表)
        沒有一個數字適合所有商品，請依據您選的標的對號入座：

        | 資產類型 | 代表標的 | 建議年數 | 原因 (市場特性) |
        | :--- | :--- | :--- | :--- |
        | **成熟大盤** | S&P 500, 台股加權 | **15 ~ 20 年** | 需涵蓋 2008 或 2020 崩盤，測試抗壓性。 |
        | **原物料** | 黃金, 白銀 (SIVR), 原油 | **20 年 (滿)** | 週期極長，往往沈睡 10 年才爆發一次。 |
        | **成長科技** | NVDA, TSLA | **5 ~ 10 年** | 產業變化快，太舊的資料參考價值較低。 |
        | **加密貨幣** | BTC, ETH | **4 ~ 8 年** | 幣圈變化極快，通常 4 年一次大循環。 |

        ---

        ### 3️⃣ 進階：如何做「壓力測試」？ (SOP)
        當您發現長短期回測結果不一致時，請這樣做：

        1.  **拉到最長 (20年)**：測「體質」。如果長期下來都輸給買進持有，直接放棄此商品。
        2.  **縮到中期 (5~8年)**：測「近況」。如果長期賺但近期賠，代表現在處於「休眠期/盤整期」。
            * **對策**：改用 **🔴 紅線 (穩健)**，或暫時觀望。
        3.  **滑動測試**：隨意拉動年份條 (如 12年、15年)。如果績效忽高忽低，代表策略不穩定 (靠運氣)。

        """)

    if run_btn:
        with st.spinner(f'正在進行高精度掃描...'):
            price_data = get_data(ticker, years)

            if price_data is None or price_data.empty:
                st.error(f"❌ 找不到代碼：{ticker}，請確認是否輸入正確 (需為 Yahoo Finance 格式)。")
            else:
                last_date = price_data.index[-1].strftime('%Y-%m-%d')
                last_price = price_data.iloc[-1]
                st.success(f"📅 資料更新至：{last_date} | 最新收盤價：{last_price:.2f} | 代碼: {ticker}")

                res = analyze_strategy(price_data)

                ret_safe = (res["safe_robust"].iloc[-1] - 1) * 100
                ret_turbo = (res["turbo_robust"].iloc[-1] - 1) * 100
                ret_hold = (res["hold"].iloc[-1] - 1) * 100

                # --- A. 指標顯示 ---
                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric("🔴 穩健策略 (紅)", f"{ret_safe:.1f}%")
                c2.metric("🔵 狂暴策略 (藍)", f"{ret_turbo:.1f}%", delta=f"{ret_turbo - ret_hold:.1f}% (vs大盤)")
                c3.metric("⚪ 買進持有 (灰)", f"{ret_hold:.1f}%")
                c4.metric("⚙️ 最佳均線", f"MA{res['win_robust']}")
                c5.metric("🌊 當前波動", f"{res['curr_vol'] * 100:.1f}%")

                # --- B. AI 操作建議卡片 ---
                st.subheader("💡 AI 操作建議 (基於藍線-狂暴策略)")
                curr_pos = res["curr_pos"]

                # 防呆邏輯：如果 TSMOM (狂暴版) 輸給 買進持有，就顯示不推薦
                if ret_turbo < ret_hold:
                    msg = f"❌ **不推薦使用此模型 (跑輸大盤)**"
                    bg_color = "rgba(255, 50, 50, 0.2)"
                    pos_text = "建議：直接買進持有 (Buy & Hold) 或更換標的"
                elif curr_pos > 0:
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
                ax.plot(res["safe_robust"], color='red', linewidth=2.0, alpha=0.9, label='紅線：穩健+控倉 (防禦)')

                ax.plot(res["safe_peak"], color='orange', linestyle=':', linewidth=1.5, alpha=0.6,
                        label='橘線：貪婪+控倉 (參考)')
                ax.plot(res["turbo_peak"], color='purple', linestyle=':', linewidth=1.5, alpha=0.6,
                        label='紫線：貪婪+狂暴 (參考)')

                ax.plot(res["hold"], color='gray', linestyle='--', linewidth=2.0, alpha=0.6,
                        label='灰線：買進持有 (對照組)')

                ax.set_title(f"策略對決：MA{res['win_robust']} (Current Best)", fontsize=14)
                ax.legend(loc='upper left', fontsize=10)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                st.warning("""
                **⚠️ 狂暴模式注意：**
                * 建議無視波動率，直接給予 **100% 滿倉** 的訊號。
                * 在牛市 (如 NVDA) 會賺非常多，但在盤整或空頭可能會面臨較大的回檔。
                """)
    else:
        st.info("👈 請在左側選單選擇股票，或輸入代碼開始分析")