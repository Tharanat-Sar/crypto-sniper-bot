import os
import json
import datetime
import requests
import ccxt
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = OpenAI(api_key=OPENAI_API_KEY)
exchange = ccxt.okx({"enableRateLimit": True})

WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "DOGE/USDT"]
TRADES_FILE = "trades.json"

# --- ระบบจัดการประวัติการเทรด (Ledger) ---
def load_trades():
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_trades(trades):
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)

def send_telegram_alert(message: str):
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

# --- Indicators ---
def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_squeeze(df):
    basis = df["close"].rolling(window=20).mean()
    dev = df["close"].rolling(window=20).std()
    upper_bb = basis + (dev * 2.0)
    lower_bb = basis - (dev * 2.0)

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

def get_ohlcv_data(symbol: str, timeframe: str, limit: int = 250):
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.iloc[:-1].copy()

# --- เช็คสถานะไม้ที่เปิดค้างไว้ (TP / SL Tracker) ---
def update_open_positions():
    trades = load_trades()
    updated = False

    for trade in trades:
        if trade.get("status") != "OPEN":
            continue

        symbol = trade["symbol"]
        try:
            ticker = exchange.fetch_ticker(symbol)
            curr_price = ticker["last"]
        except Exception:
            continue

        entry = trade["entry"]
        tp = trade["tp"]
        sl = trade["sl"]

        # ชน TP (+1.5%)
        if curr_price >= tp:
            trade["status"] = "WIN"
            trade["exit_price"] = curr_price
            trade["closed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            trade["pnl_pct"] = round(((curr_price - entry) / entry) * 100, 2)
            updated = True
            msg = (
                f"🎉 *[PAPER TRADE: TAKE PROFIT]*\n"
                f"เหรียญ: `{symbol}`\n"
                f"สถานะ: 🟢 WIN (+{trade['pnl_pct']}%)\n"
                f"ราคาปิด: `{curr_price}` (เป้า `{tp}`)"
            )
            send_telegram_alert(msg)

        # ชน SL (-1.0%)
        elif curr_price <= sl:
            trade["status"] = "LOSS"
            trade["exit_price"] = curr_price
            trade["closed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            trade["pnl_pct"] = round(((curr_price - entry) / entry) * 100, 2)
            updated = True
            msg = (
                f"🛡️ *[PAPER TRADE: STOP LOSS]*\n"
                f"เหรียญ: `{symbol}`\n"
                f"สถานะ: 🔴 LOSS ({trade['pnl_pct']}%)\n"
                f"ราคาปิด: `{curr_price}` (เป้า `{sl}`)"
            )
            send_telegram_alert(msg)

    if updated:
        save_trades(trades)

# --- ระบบสรุปผลรายงาน (Daily / Weekly / Monthly Report) ---
def generate_summary_reports():
    trades = load_trades()
    if not trades:
        return

    # เวลาไทย (UTC + 7)
    now_th = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    today_str = now_th.strftime("%Y-%m-%d")

    # สรุปทุกรอบ 23:45 น. ของวัน
    if now_th.hour == 23 and now_th.minute >= 40:
        closed_today = [t for t in trades if t.get("status") in ["WIN", "LOSS"] and t.get("closed_at", "").startswith(today_str)]
        wins = [t for t in closed_today if t["status"] == "WIN"]
        losses = [t for t in closed_today if t["status"] == "LOSS"]
        total_closed = len(closed_today)
        win_rate = (len(wins) / total_closed * 100) if total_closed > 0 else 0
        total_pnl = sum([t.get("pnl_pct", 0) for t in closed_today])

        daily_msg = (
            f"📋 *[DAILY SUMMARY REPORT - {today_str}]*\n"
            f"------------------------------------\n"
            f"🎯 ไม้ที่ปิดวันนี้: `{total_closed}` ไม้\n"
            f"🟢 ชนะ (Win): `{len(wins)}` | 🔴 แพ้ (Loss): `{len(losses)}`\n"
            f"🏆 Win Rate วันนี้: *{win_rate:.1f}%*\n"
            f"📈 กำไรรวมวันนี้ (PnL): *{total_pnl:+.2f}%*\n"
            f"------------------------------------"
        )
        send_telegram_alert(daily_msg)

    # สรุปประจำสัปดาห์ (ทุกคืนวันอาทิตย์ รอบ 23:45 น.)
    if now_th.weekday() == 6 and now_th.hour == 23 and now_th.minute >= 40:
        all_closed = [t for t in trades if t.get("status") in ["WIN", "LOSS"]]
        w = len([t for t in all_closed if t["status"] == "WIN"])
        l = len([t for t in all_closed if t["status"] == "LOSS"])
        wr = (w / len(all_closed) * 100) if all_closed else 0
        cum_pnl = sum([t.get("pnl_pct", 0) for t in all_closed])

        weekly_msg = (
            f"📊 *[WEEKLY PERFORMANCE REPORT]*\n"
            f"------------------------------------\n"
            f"📦 ออเดอร์สะสมทั้งหมด: `{len(all_closed)}` ไม้\n"
            f"🎯 Win Rate สะสมรวม: *{wr:.1f}%*\n"
            f"💰 PnL สุทธิสะสม: *{cum_pnl:+.2f}%*\n"
            f"------------------------------------"
        )
        send_telegram_alert(weekly_msg)

def scan_symbol(symbol: str):
    print(f"\n🔍 กำลังตรวจสอบ {symbol}...")

    # ตรวจสอบว่ามีไม้ของเหรียญนี้เปิดค้างอยู่หรือไม่ (เปิดได้ทีละ 1 ไม้ต่อ 1 เหรียญ)
    trades = load_trades()
    for t in trades:
        if t.get("symbol") == symbol and t.get("status") == "OPEN":
            print(f"⏸️ ข้าม: มีไม้ {symbol} ถือเปิดค้างรอ TP/SL อยู่แล้ว")
            return

    # ด่านที่ 1: TF 1h
    df_1h = get_ohlcv_data(symbol, "1h", limit=250)
    df_1h["ema200"] = calculate_ema(df_1h["close"], 200)
    last_1h = df_1h.iloc[-1]
    close_1h = last_1h["close"]
    ema200_1h = last_1h["ema200"]

    if close_1h <= ema200_1h:
        print(f"⏭️ ปัดตก: แท่ง 1h ({close_1h}) อยู่ใต้ EMA 200 ({ema200_1h:.2f})")
        return

    # ด่านที่ 2: TF 15m
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

    # ด่านที่ 3: ให้ AI วิเคราะห์ความเสี่ยง
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

    # ด่านที่ 4: บันทึก Paper Trade และยิงแจ้งเตือน
    if decision == "PASS":
        entry = decision_data["entry"]
        sl = decision_data["sl"]
        tp = decision_data["tp"]

        # บันทึกไม้ใหม่เข้า Ledger
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        new_trade = {
            "id": int(datetime.datetime.now().timestamp()),
            "symbol": symbol,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "status": "OPEN",
            "opened_at": now_str,
            "closed_at": None,
            "pnl_pct": None,
            "ai_reason": reason
        }
        trades.append(new_trade)
        save_trades(trades)

        msg = (
            f"🎯 *[SNIPER SIGNAL: {symbol}]*\n"
            f"-----------------------------\n"
            f"📍 *ราคาเข้า (Entry):* `{entry}`\n"
            f"🛡️ *ตัดขาดทุน (SL -1.0%):* `{sl}`\n"
            f"💰 *ทำกำไร (TP +1.5%):* `{tp}`\n"
            f"-----------------------------\n"
            f"📊 *Indicators:* RSI {rsi_15m:.1f} | Gap {gap_pct_15m:.2f}%\n"
            f"⚡ *Squeeze:* `{squeeze_status}`\n"
            f"💡 *มุมมอง AI:* {reason}\n"
            f"📝 *โหมด Paper Trading: บันทึกไม้เข้า Ledger แล้ว*"
        )
        send_telegram_alert(msg)

def run():
    print("🚀 เริ่มต้นระบบสไนเปอร์...")
    # 1. เช็คไม้ที่ถือค้างไว้ก่อนว่าชน TP / SL หรือยัง
    update_open_positions()

    # 2. สแกนหาไม้ใหม่
    for sym in WATCHLIST:
        try:
            scan_symbol(sym)
        except Exception as e:
            print(f"⚠️ มีข้อผิดพลาดในการสแกน {sym}: {e}")

    # 3. ตรวจสอบเงื่อนไขการส่งรายงานสรุปยอดประจำวัน/สัปดาห์
    generate_summary_reports()

if __name__ == "__main__":
    run()