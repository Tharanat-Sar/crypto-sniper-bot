import os
import json
import requests
import ccxt
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# โหลด Environment Variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = OpenAI(api_key=OPENAI_API_KEY)
# เปลี่ยนเป็น OKX เพื่อให้ GitHub Actions (US Server) ดึงข้อมูลตลาดได้โดยไม่ติดบล็อก HTTP 451
exchange = ccxt.okx({"enableRateLimit": True})

# 5 เหรียญหลักตาม Sniper Playbook
WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "DOGE/USDT"]

# --- ฟังก์ชันคำนวณ Indicators (Pure Pandas) ---
def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_squeeze(df):
    """
    คำนวณ Volatility Squeeze (Bollinger Bands vs Keltner Channels):
    - SQUEEZE_ON: กรอบ BB บีบตัวอยู่ใน KC (ช่วงสะสมพลัง ราคายังไม่เลือกทาง)
    - SQUEEZE_FIRE: กรอบ BB ขยายตัวทะลุ KC ออกมา (มีโมเมนตัมระเบิดตัว)
    """
    # 1. Bollinger Bands (20, 2)
    basis = df["close"].rolling(window=20).mean()
    dev = df["close"].rolling(window=20).std()
    upper_bb = basis + (dev * 2.0)
    lower_bb = basis - (dev * 2.0)

    # 2. Keltner Channels (20, 1.5 ATR)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20).mean()

    upper_kc = basis + (atr * 1.5)
    lower_kc = basis - (atr * 1.5)

    last_idx = df.index[-1]
    is_squeeze_on = (lower_bb.loc[last_idx] > lower_kc.loc[last_idx]) and (upper_bb.loc[last_idx] < upper_kc.loc[last_idx])
    return "SQUEEZE_ON" if is_squeeze_on else "SQUEEZE_FIRE"

def send_telegram_alert(message: str):
    """ส่งข้อความเข้า Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

def get_ohlcv_data(symbol: str, timeframe: str, limit: int = 250):
    """ดึงแท่งเทียนและตัดแท่งปัจจุบันออกเพื่อป้องกัน Repaint"""
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.iloc[:-1].copy()

def scan_symbol(symbol: str):
    print(f"\n🔍 กำลังตรวจสอบ {symbol}...")

    # --- ด่านที่ 1: ตรวจสอบแนวโน้มภาพใหญ่ 1 Hour ---
    df_1h = get_ohlcv_data(symbol, "1h", limit=250)
    df_1h["ema200"] = calculate_ema(df_1h["close"], 200)
    
    last_1h = df_1h.iloc[-1]
    close_1h = last_1h["close"]
    ema200_1h = last_1h["ema200"]

    if close_1h <= ema200_1h:
        print(f"⏭️ ปัดตก: แท่ง 1h ({close_1h}) อยู่ใต้ EMA 200 ({ema200_1h:.2f})")
        return

    # --- ด่านที่ 2: คำนวณจังหวะสับไก 15 Minutes ---
    df_15m = get_ohlcv_data(symbol, "15m", limit=250)
    df_15m["ema200"] = calculate_ema(df_15m["close"], 200)
    df_15m["ema50"] = calculate_ema(df_15m["close"], 50)
    df_15m["rsi"] = calculate_rsi(df_15m["close"], 14)
    squeeze_status = calculate_squeeze(df_15m)

    last_15m = df_15m.iloc[-1]
    close_15m = last_15m["close"]
    ema200_15m = last_15m["ema200"]
    ema50_15m = last_15m["ema50"]
    rsi_15m = last_15m["rsi"]

    gap_pct_15m = ((close_15m - ema200_15m) / ema200_15m) * 100
    avg_vol = df_15m["volume"].tail(5).mean()
    vol_ratio = last_15m["volume"] / avg_vol if avg_vol > 0 else 1.0

    market_snapshot = {
        "symbol": symbol,
        "tf_1h_trend": "UPTREND (Above EMA200)",
        "close_15m": round(close_15m, 4),
        "ema200_15m": round(ema200_15m, 4),
        "ema50_15m": round(ema50_15m, 4),
        "ema_gap_pct": round(gap_pct_15m, 2),
        "rsi_15m": round(rsi_15m, 2),
        "vol_ratio_vs_5avg": round(vol_ratio, 2),
        "squeeze_status": squeeze_status
    }

    # --- ด่านที่ 3: ให้ AI วิเคราะห์ความเสี่ยง ---
    prompt = f"""
    คุณเป็น AI Risk Manager คัดกรองสัญญาณซื้อ Day Trade (TF 15m) สำหรับคู่ Spot เป้าหมายคือ Win Rate 60%
    กฎเหล็ก:
    1. ราคาต้องยืนเหนือ EMA 200 ทั้งใน 1h และ 15m
    2. หนังสติ๊กต้องไม่ตึง: gap_pct บน 15m ต้องอยู่ระหว่าง 0.1% ถึง 1.0% เท่านั้น (ถ้าเกิน 1.0% = ห้ามเข้า เสี่ยงย่อตัว)
    3. RSI ต้องอยู่ช่วง 50 ถึง 65 (ถ้าเกิน 68 = ห้ามเข้า Overbought เสี่ยงดอย)
    4. Volatility Squeeze: หาก squeeze_status คือ SQUEEZE_FIRE จะมีน้ำหนักความมั่นใจสูงขึ้น
    
    ข้อมูลกราฟล่าสุด:
    {json.dumps(market_snapshot, indent=2)}

    จงวิเคราะห์แล้วตอบกลับเป็น JSON format เท่านั้น:
    {{
      "decision": "PASS" หรือ "REJECT",
      "reason": "เหตุผลสั้นๆ ไม่เกิน 15 คำ",
      "entry": {close_15m},
      "sl": {round(close_15m * 0.99, 4)},
      "tp": {round(close_15m * 1.015, 4)}
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    decision_data = json.loads(response.choices[0].message.content)
    decision = decision_data.get("decision", "REJECT")
    reason = decision_data.get("reason", "")
    print(f"ผลประเมิน {symbol}: {decision} ({reason})")

    # --- ด่านที่ 4: ยิงแจ้งเตือนเข้า Telegram ---
    if decision == "PASS":
        entry = decision_data["entry"]
        sl = decision_data["sl"]
        tp = decision_data["tp"]

        msg = (
            f"🎯 *[SNIPER SIGNAL: {symbol}]*\n"
            f"-----------------------------\n"
            f"📍 *ราคาเข้า (Entry):* `{entry}`\n"
            f"🛡️ *ตัดขาดทุน (SL -1.0%):* `{sl}`\n"
            f"💰 *ทำกำไร (TP +1.5%):* `{tp}`\n"
            f"-----------------------------\n"
            f"📊 *Indicators:* RSI {rsi_15m:.1f} | Gap {gap_pct_15m:.2f}%\n"
            f"⚡ *Squeeze:* `{squeeze_status}`\n"
            f"💡 *มุมมอง AI:* {reason}"
        )
        send_telegram_alert(msg)

def run():
    print("🚀 เริ่มต้นสแกน 5 เหรียญสไนเปอร์...")
    for sym in WATCHLIST:
        try:
            scan_symbol(sym)
        except Exception as e:
            print(f"⚠️ มีข้อผิดพลาดในการสแกน {sym}: {e}")

if __name__ == "__main__":
    run()