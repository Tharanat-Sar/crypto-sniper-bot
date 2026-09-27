import os
import json
import time
import datetime
import requests
import ccxt
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# --- โหลด Environment Variables ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = OpenAI(api_key=OPENAI_API_KEY)
exchange = ccxt.okx({"enableRateLimit": True})

# --- ตั้งค่ากลยุทธ์และพารามิเตอร์การควบคุมความเสี่ยง ---
WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "DOGE/USDT"]
TRADES_FILE = "trades.json"
STATE_FILE = "state.json"

# กฎการคุมความเสี่ยง (Risk Rules)
MAX_OPEN_POSITIONS = 2      # จำกัดถือพร้อมกันได้สูงสุดไม่เกิน 2 ไม้
MAX_DAILY_LOSSES = 2        # Daily Circuit Breaker: แพ้สะสมครบ 2 ไม้ในวันเดียว หยุดเปิดไม้ใหม่ทันที
COOLDOWN_HOURS = 2          # Post-Loss Cooldown: พักเหรียญที่เพิ่งแพ้ 2 ชั่วโมง
TIME_STOP_HOURS = 24        # Time Stop: ถือแช่นานเกิน 24 ชม. บังคับปิดตลาดคืนเงินสด
TRAILING_TRIGGER_PCT = 1.5  # แตะ +1.5% เปิดโหมด Trailing Stop (ไม่ขายหมู)
TRAILING_STEP_PCT = 0.5     # ขยับเส้นขายตามหลังราคาสูงสุด 0.5%
BREAKEVEN_TRIGGER_PCT = 0.8 # แตะ +0.8% ขยับ SL มากันหน้าทุน (Entry Price)

# --- ระบบจัดการประวัติและสถานะระบบ (Ledger & State) ---
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

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_daily_report": "", "last_weekly_report": ""}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_daily_report": "", "last_weekly_report": ""}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def send_telegram_alert(message: str):
    """ส่งข้อความเข้า Telegram พร้อมเว้นระยะป้องกัน Rate Limit (HTTP 429)"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
        time.sleep(1.5)  # เว้น 1.5 วินาที ป้องกัน Telegram บล็อกหรือการ์ดตกหล่น
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# --- ฟังก์ชันคำนวณทางเทคนิค (Pure Pandas) ---
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

# --- เช็คสถานะไม้: Trailing Stop / Breakeven / Time Stop / Soft SL ---
def update_open_positions():
    trades = load_trades()
    updated = False
    now = datetime.datetime.now()

    for trade in trades:
        if trade.get("status") != "OPEN":
            continue

        symbol = trade["symbol"]
        try:
            ticker = exchange.fetch_ticker(symbol)
            curr_price = float(ticker["last"])
        except Exception:
            continue

        entry = float(trade["entry"])
        sl = float(trade["sl"])
        opened_at_str = trade.get("opened_at")
        try:
            opened_at = datetime.datetime.strptime(opened_at_str, "%Y-%m-%d %H:%M") if opened_at_str else now
        except Exception:
            opened_at = now

        # อัปเดตราคาสูงสุดที่ไม้เคยทำได้ (Highest Price Tracking)
        highest_price = max(trade.get("highest_price", entry), curr_price)
        trade["highest_price"] = highest_price

        current_pnl = ((curr_price - entry) / entry) * 100
        highest_pnl = ((highest_price - entry) / entry) * 100

        # 1. กลไก Time Stop (ครบ 24 ชม. ปิดตลาดเคลียร์เงินสด)
        hours_held = (now - opened_at).total_seconds() / 3600
        if hours_held >= TIME_STOP_HOURS:
            trade["status"] = "TIME_STOP"
            trade["exit_price"] = curr_price
            trade["closed_at"] = now.strftime("%Y-%m-%d %H:%M")
            trade["pnl_pct"] = round(current_pnl, 2)
            updated = True
            msg = (
                f"⏱️ *[PAPER TRADE: 24H TIME STOP]*\n"
                f"เหรียญ: `{symbol}` (ถือครบ {hours_held:.1f} ชม.)\n"
                f"สถานะ: ปิดคืนทุน/ตัดขาดทุนตลาด ({trade['pnl_pct']:+.2f}%)\n"
                f"ราคาปิด: `{curr_price}`"
            )
            send_telegram_alert(msg)
            continue

        # 2. กลไก Trailing Stop (เมื่อกำไรพุ่งเกิน +1.5% ไม่รีบขาย ปล่อยให้รันต่อ)
        if highest_pnl >= TRAILING_TRIGGER_PCT:
            trade["trailing_active"] = True
            trail_sl_pct = highest_pnl - TRAILING_STEP_PCT
            trail_sl_price = entry * (1 + (trail_sl_pct / 100))

            # อัปเดต SL ตามหลังราคาขึ้นไป
            if trail_sl_price > trade["sl"]:
                trade["sl"] = trail_sl_price

            # หากราคาย่อลงมาแตะเส้น Trailing ให้ขายทำกำไร
            if curr_price <= trail_sl_price:
                trade["status"] = "WIN"
                trade["exit_price"] = curr_price
                trade["closed_at"] = now.strftime("%Y-%m-%d %H:%M")
                trade["pnl_pct"] = round(current_pnl, 2)
                updated = True
                msg = (
                    f"🚀 *[PAPER TRADE: TRAILING TAKE PROFIT]*\n"
                    f"เหรียญ: `{symbol}`\n"
                    f"สถานะ: 🟢 WIN (+{trade['pnl_pct']}%)\n"
                    f"ราคาปิด: `{curr_price}` (จุดสูงสุดเคยแตะ +{highest_pnl:.2f}%)"
                )
                send_telegram_alert(msg)
                continue

        # 3. กลไก Breakeven Stop (กำไรเคยแตะ +0.8% ขยับ SL มากันหน้าทุน)
        elif highest_pnl >= BREAKEVEN_TRIGGER_PCT and not trade.get("is_breakeven"):
            trade["sl"] = entry
            trade["is_breakeven"] = True
            updated = True
            msg = (
                f"🛡️ *[BREAKEVEN ACTIVATED: {symbol}]*\n"
                f"ราคาขึ้นแตะ `+{highest_pnl:.2f}%` ระบบขยับ Stop Loss มาล็อกหน้าทุน `{entry}` เรียบร้อยแล้ว (การันตีไม่แพ้)"
            )
            send_telegram_alert(msg)

        # 4. กลไก Stop Loss ตามปกติ (หรือชน Breakeven ที่ขยับมา)
        if curr_price <= trade["sl"]:
            status = "WIN" if current_pnl >= 0 else "LOSS"
            icon = "🟢" if status == "WIN" else "🔴"
            trade["status"] = status
            trade["exit_price"] = curr_price
            trade["closed_at"] = now.strftime("%Y-%m-%d %H:%M")
            trade["pnl_pct"] = round(current_pnl, 2)
            updated = True
            msg = (
                f"🛡️ *[PAPER TRADE: STOP LOSS]*\n"
                f"เหรียญ: `{symbol}`\n"
                f"สถานะ: {icon} {status} ({trade['pnl_pct']:+.2f}%)\n"
                f"ราคาปิด: `{curr_price}` (เป้า SL `{trade['sl']}`)"
            )
            send_telegram_alert(msg)

    if updated:
        save_trades(trades)

# --- ระบบสรุปผลรายงาน (Daily & Weekly แบบไม่ส่งซ้ำ ป้องกัน NoneType Crash) ---
def generate_summary_reports():
    trades = load_trades()
    state = load_state()
    now_th = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    today_str = now_th.strftime("%Y-%m-%d")
    week_str = now_th.strftime("%Y-W%U")

    # ส่งสรุปประจำวันรอบตี 3 (03:00 น.) เพียงครั้งเดียวต่อวัน
    if now_th.hour == 3 and state.get("last_daily_report") != today_str:
        # ป้องกัน NoneType ด้วย (t.get(...) or "")
        closed_today = [t for t in trades if (t.get("closed_at") or "").startswith(today_str)]
        opened_today = [t for t in trades if (t.get("opened_at") or "").startswith(today_str)]
        wins = [t for t in closed_today if t.get("status") == "WIN"]
        losses = [t for t in closed_today if t.get("status") in ["LOSS", "TIME_STOP"]]
        total_closed = len(closed_today)
        win_rate = (len(wins) / total_closed * 100) if total_closed > 0 else 0.0
        realized_pnl = sum([(t.get("pnl_pct") or 0) for t in closed_today])

        open_positions = [t for t in trades if t.get("status") == "OPEN"]
        open_text = ""
        if open_positions:
            open_text = "\n\n⏳ *[ไม้ที่ถือข้ามวัน (Holding)]:*\n"
            for op in open_positions:
                sym = op["symbol"]
                try:
                    curr = float(exchange.fetch_ticker(sym)["last"])
                    entry_val = float(op["entry"])
                    unrealized = ((curr - entry_val) / entry_val) * 100
                    pnl_icon = "🟢" if unrealized >= 0 else "🔴"
                    open_text += f"• `{sym}`: เข้า `{entry_val}` | ล่าสุด `{curr}` ({pnl_icon} {unrealized:+.2f}%)\n"
                except Exception:
                    open_text += f"• `{sym}`: เข้า `{op.get('entry')}` (รอเช็คราคา)\n"
        else:
            open_text = "\n\n⏳ *ไม้ที่ถือค้างอยู่:* ไม่มี (พอร์ตว่าง 100%)"

        daily_msg = (
            f"📋 *[DAILY SUMMARY REPORT - {today_str}]*\n"
            f"------------------------------------\n"
            f"🎯 ออเดอร์เปิดวันนี้: `{len(opened_today)}` ไม้ | ปิดวันนี้: `{total_closed}` ไม้\n"
            f"🟢 ชนะ: `{len(wins)}` | 🔴 แพ้/Time Stop: `{len(losses)}`\n"
            f"🏆 Win Rate: *{win_rate:.1f}%*\n"
            f"📈 Realized PnL รวม: *{realized_pnl:+.2f}%*"
            f"{open_text}\n"
            f"------------------------------------"
        )
        send_telegram_alert(daily_msg)
        state["last_daily_report"] = today_str
        save_state(state)

    # ส่งสรุปประจำสัปดาห์ (เช้าวันจันทร์ เวลา 03:00 น.)
    if now_th.weekday() == 0 and now_th.hour == 3 and state.get("last_weekly_report") != week_str:
        all_closed = [t for t in trades if t.get("status") in ["WIN", "LOSS", "TIME_STOP"]]
        w = len([t for t in all_closed if t.get("status") == "WIN"])
        wr = (w / len(all_closed) * 100) if all_closed else 0.0
        cum_pnl = sum([(t.get("pnl_pct") or 0) for t in all_closed])

        weekly_msg = (
            f"📊 *[WEEKLY PERFORMANCE REPORT]*\n"
            f"------------------------------------\n"
            f"📦 ไม้ที่ปิดรอบสะสม: `{len(all_closed)}` ไม้\n"
            f"🎯 Win Rate รวม: *{wr:.1f}%*\n"
            f"💰 Net Realized PnL: *{cum_pnl:+.2f}%*\n"
            f"------------------------------------"
        )
        send_telegram_alert(weekly_msg)
        state["last_weekly_report"] = week_str
        save_state(state)

# --- ระบบสแกนและคัดกรองสัญญาณซื้อ (Scanner) ---
def scan_symbol(symbol: str):
    print(f"\n🔍 กำลังตรวจสอบ {symbol}...")
    trades = load_trades()
    now = datetime.datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # 1. เช็ค Daily Circuit Breaker (ถ้าแพ้วันนี้ครบ 2 ไม้ หยุดทันที)
    losses_today = [
        t for t in trades 
        if t.get("status") in ["LOSS", "TIME_STOP"] 
        and (t.get("closed_at") or "").startswith(today_str)
        and (t.get("pnl_pct") or 0) < 0
    ]
    if len(losses_today) >= MAX_DAILY_LOSSES:
        print(f"🚨 Circuit Breaker ทำงาน: วันนี้แพ้ครบ {len(losses_today)} ไม้แล้ว งดเข้าซื้อใหม่")
        return

    # 2. เช็ค Max Open Positions (ถือพร้อมกันได้สูงสุด 2 ไม้)
    open_positions = [t for t in trades if t.get("status") == "OPEN"]
    if len(open_positions) >= MAX_OPEN_POSITIONS:
        print(f"⏸️ พอร์ตเต็ม: เปิดครบ {MAX_OPEN_POSITIONS} ไม้แล้ว")
        return

    # 3. เช็คว่ามีเหรียญนี้เปิดค้างอยู่หรือไม่
    if any(t.get("symbol") == symbol and t.get("status") == "OPEN" for t in trades):
        print(f"⏸️ ข้าม: มีไม้ {symbol} เปิดค้างรอ TP/SL อยู่แล้ว")
        return

    # 4. เช็ค Post-Loss Cooldown (ห้ามเข้าเหรียญเดิมที่เพิ่งแพ้ภายใน 2 ชม.)
    recent_losses = [
        t for t in trades 
        if t.get("symbol") == symbol 
        and t.get("status") in ["LOSS", "TIME_STOP"] 
        and t.get("closed_at")
    ]
    if recent_losses:
        try:
            last_loss = max(recent_losses, key=lambda x: str(x.get("closed_at") or ""))
            closed_time = datetime.datetime.strptime(last_loss["closed_at"], "%Y-%m-%d %H:%M")
            hours_passed = (now - closed_time).total_seconds() / 3600
            if hours_passed < COOLDOWN_HOURS:
                print(f"⏳ ข้าม {symbol}: อยู่ในช่วง Cooldown หลังแพ้ (ผ่านไป {hours_passed:.1f}/{COOLDOWN_HOURS} ชม.)")
                return
        except Exception as e:
            print(f"⚠️ Cooldown check error for {symbol}: {e}")

    # ด่านเทคนิคที่ 1: TF 1h
    df_1h = get_ohlcv_data(symbol, "1h", limit=250)
    df_1h["ema200"] = calculate_ema(df_1h["close"], 200)
    last_1h = df_1h.iloc[-1]
    if last_1h["close"] <= last_1h["ema200"]:
        print(f"⏭️ ปัดตก: แท่ง 1h ({last_1h['close']}) อยู่ใต้ EMA 200")
        return

    # ด่านเทคนิคที่ 2: TF 15m
    df_15m = get_ohlcv_data(symbol, "15m", limit=250)
    df_15m["ema200"] = calculate_ema(df_15m["close"], 200)
    df_15m["ema50"] = calculate_ema(df_15m["close"], 50)
    df_15m["rsi"] = calculate_rsi(df_15m["close"], 14)
    squeeze_status = calculate_squeeze(df_15m)

    last_15m = df_15m.iloc[-1]
    close_15m = float(last_15m["close"])
    ema200_15m = float(last_15m["ema200"])
    rsi_15m = float(last_15m["rsi"])
    gap_pct_15m = ((close_15m - ema200_15m) / ema200_15m) * 100

    market_snapshot = {
        "symbol": symbol,
        "tf_1h_trend": "UPTREND (Above EMA200)",
        "close_15m": round(close_15m, 4),
        "ema200_15m": round(ema200_15m, 4),
        "ema_gap_pct": round(gap_pct_15m, 2),
        "rsi_15m": round(rsi_15m, 2),
        "squeeze_status": squeeze_status
    }

    # ด่านที่ 3: OpenAI Risk Filter (ล็อกภาษาไทยล้วน)
    prompt = f"""
    คุณเป็น AI Risk Manager คัดกรองสัญญาณซื้อ Day Trade (TF 15m) สำหรับคู่ Spot เป้าหมาย Win Rate 60%
    กฎเหล็ก:
    1. ราคาต้องยืนเหนือ EMA 200 ทั้ง 1h และ 15m
    2. EMA Gap 15m ต้องอยู่ระหว่าง 0.1% ถึง 1.0%
    3. RSI ต้องอยู่ระหว่าง 50 ถึง 65 (ห้ามเกิน 68)
    4. Squeeze Fire จะให้น้ำหนักสูงขึ้น

    ข้อมูลตลาด:
    {json.dumps(market_snapshot, indent=2)}

    ตอบกลับเป็น JSON เท่านั้น และเหตุผล (reason) ต้องเป็นภาษาไทยล้วนไม่เกิน 15 คำ:
    {{
      "decision": "PASS" หรือ "REJECT",
      "reason": "เหตุผลสั้นๆ ภาษาไทยล้วน",
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

    # ด่านที่ 4: บันทึกออเดอร์และแจ้งเตือน
    if decision == "PASS":
        entry = decision_data["entry"]
        new_trade = {
            "id": int(datetime.datetime.now().timestamp()),
            "symbol": symbol,
            "entry": entry,
            "highest_price": entry,
            "sl": decision_data["sl"],
            "tp": decision_data["tp"],
            "status": "OPEN",
            "opened_at": now.strftime("%Y-%m-%d %H:%M"),
            "closed_at": None,
            "pnl_pct": None,
            "is_breakeven": False,
            "trailing_active": False,
            "ai_reason": reason
        }
        trades.append(new_trade)
        save_trades(trades)

        msg = (
            f"🎯 *[SNIPER SIGNAL: {symbol}]*\n"
            f"-----------------------------\n"
            f"📍 *ราคาเข้า (Entry):* `{entry}`\n"
            f"🛡️ *ตัดขาดทุน (SL -1.0%):* `{new_trade['sl']}`\n"
            f"💰 *เป้ากำไรแรก (TP +1.5%):* `{new_trade['tp']}`\n"
            f"-----------------------------\n"
            f"📊 *Indicators:* RSI {rsi_15m:.1f} | Gap {gap_pct_15m:.2f}%\n"
            f"⚡ *Squeeze:* `{squeeze_status}`\n"
            f"💡 *มุมมอง AI:* {reason}\n"
            f"📝 *โหมด Paper Trading: บันทึกไม้เข้า Ledger แล้ว*"
        )
        send_telegram_alert(msg)

def run():
    print(f"🚀 เริ่มต้นระบบสไนเปอร์ [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]...")
    # 1. เช็คสถานะไม้ค้างเดิม (Trailing, Breakeven, SL, 24h Time Stop)
    update_open_positions()

    # 2. วนลูปสแกนหาไม้ใหม่
    for sym in WATCHLIST:
        try:
            scan_symbol(sym)
        except Exception as e:
            print(f"⚠️ ข้อผิดพลาดในการสแกน {sym}: {e}")

    # 3. ตรวจสอบการส่งรายงาน Daily / Weekly
    generate_summary_reports()

if __name__ == "__main__":
    run()