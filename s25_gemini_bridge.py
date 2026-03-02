"""
S25 LUMIERE - Gemini Bridge  v1.0
Pont Windows: polling HA input_text.ai_prompt -> Gemini -> input_text.ai_response + ai_analysis.json GitHub
Usage: python s25_gemini_bridge.py
Config: s25_config.py (local, jamais pushé sur GitHub)
"""
import urllib.request
import urllib.error
import json
import base64
import time
import re
import sys
import os
from datetime import datetime, timezone

# ─── LOAD CONFIG ──────────────────────────────────────────
try:
    import s25_config as CFG
    HA_URL       = getattr(CFG, 'HA_URL',       'http://homeassistant.local:8123')
    HA_TOKEN     = getattr(CFG, 'HA_TOKEN',     '')
    GEMINI_AGENT = getattr(CFG, 'GEMINI_AGENT', '')
    GH_PAT       = getattr(CFG, 'GH_PAT',       '')
    GH_REPO      = getattr(CFG, 'GH_REPO',      'stephaneboss/arkon5-xiaomi-robotics')
    GH_FILE      = getattr(CFG, 'GH_FILE',      'ai_analysis.json')
    POLL_SEC     = getattr(CFG, 'POLL_SEC',     10)
except ImportError:
    # Fallback: env vars
    HA_URL       = os.environ.get('HA_URL',       'http://homeassistant.local:8123')
    HA_TOKEN     = os.environ.get('HA_TOKEN',     '')
    GEMINI_AGENT = os.environ.get('GEMINI_AGENT', '')
    GH_PAT       = os.environ.get('GH_PAT',       '')
    GH_REPO      = os.environ.get('GH_REPO',      'stephaneboss/arkon5-xiaomi-robotics')
    GH_FILE      = os.environ.get('GH_FILE',      'ai_analysis.json')
    POLL_SEC     = int(os.environ.get('POLL_SEC', '10'))

if not HA_TOKEN:
    print("[ERROR] HA_TOKEN manquant. Cree s25_config.py ou set env var HA_TOKEN")
    sys.exit(1)
# ──────────────────────────────────────────────────────────

HA_HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json",
}

GH_HEADERS = {
    "Authorization": f"token {GH_PAT}",
    "Accept": "application/vnd.github.v3+json",
    "Content-Type": "application/json",
    "User-Agent": "S25-Gemini-Bridge",
}

last_prompt = None


def ha_get(entity_id):
    url = f"{HA_URL}/api/states/{entity_id}"
    req = urllib.request.Request(url, headers=HA_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"[HA GET] {entity_id} error: {e}")
        return None


def ha_set(entity_id, value):
    url = f"{HA_URL}/api/services/input_text/set_value"
    body = json.dumps({"entity_id": entity_id, "value": str(value)[:255]}).encode()
    req = urllib.request.Request(url, data=body, headers=HA_HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.getcode()
    except Exception as e:
        print(f"[HA SET] {entity_id} error: {e}")
        return None


def ha_gemini(prompt_text):
    url = f"{HA_URL}/api/services/google_generative_ai_conversation/generate_content?return_response"
    body = json.dumps({
        "agent_id": GEMINI_AGENT,
        "prompt": prompt_text,
    }).encode()
    req = urllib.request.Request(url, data=body, headers=HA_HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
            return data.get("service_response", {}).get("text", "")
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:300]
        print(f"[GEMINI] HTTP {e.code}: {err}")
        return None
    except Exception as e:
        print(f"[GEMINI] Error: {e}")
        return None


def parse_gemini_signal(raw, signal_text):
    """Parse Gemini JSON response or text into trading signal dict."""
    result = {
        "action": "HOLD",
        "confidence": 0.5,
        "take_profit": None,
        "stop_loss": None,
        "symbol": "CRYPTO",
        "reason": raw[:200] if raw else "no response",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "gemini-via-ha",
        "pipeline": "ARKON-5 S25 Lumiere",
        "kimi_signal": signal_text[:200],
    }

    # Try JSON parse first (strip markdown fences if present)
    clean = re.sub(r'^```[a-z]*\n?', '', raw.strip(), flags=re.M)
    clean = re.sub(r'\n?```$', '', clean.strip(), flags=re.M)
    try:
        j = json.loads(clean)
        result["action"]      = j.get("action",      result["action"]).upper()
        result["confidence"]  = float(j.get("confidence", result["confidence"]))
        result["take_profit"] = j.get("take_profit") or j.get("tp")
        result["stop_loss"]   = j.get("stop_loss")   or j.get("sl")
        result["symbol"]      = j.get("pair") or j.get("symbol") or result["symbol"]
        result["reason"]      = j.get("reason", result["reason"])
        return result
    except Exception:
        pass

    # Fallback: regex parse
    if m := re.search(r'"?action"?\s*[=:]\s*"?(BUY|SELL|HOLD)"?', raw, re.I):
        result["action"] = m.group(1).upper()
    if m := re.search(r'"?confidence"?\s*[=:]\s*([0-9.]+)', raw, re.I):
        c = float(m.group(1))
        result["confidence"] = c if c <= 1.0 else c / 100.0
    if m := re.search(r'"?(?:take_profit|tp)"?\s*[=:]\s*([0-9.]+)', raw, re.I):
        result["take_profit"] = float(m.group(1))
    if m := re.search(r'"?(?:stop_loss|sl)"?\s*[=:]\s*([0-9.]+)', raw, re.I):
        result["stop_loss"] = float(m.group(1))
    if m := re.search(r'"?(?:pair|symbol)"?\s*[=:]\s*"?([A-Z]{2,10}/[A-Z]{3,5})"?', raw, re.I):
        result["symbol"] = m.group(1).upper()
    if m := re.search(r'"?reason"?\s*[=:]\s*"([^"]+)"', raw, re.I):
        result["reason"] = m.group(1)

    return result


def push_github(content_dict):
    """Push ai_analysis.json to GitHub."""
    if not GH_PAT:
        print("  [GH] Pas de PAT configure - skip GitHub push")
        return 0

    url = f"https://api.github.com/repos/{GH_REPO}/contents/{GH_FILE}"
    content_str = json.dumps(content_dict, ensure_ascii=False, indent=2)
    content_b64 = base64.b64encode(content_str.encode("utf-8")).decode()

    sha = None
    try:
        req = urllib.request.Request(url, headers=GH_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as r:
            sha = json.loads(r.read()).get("sha")
    except Exception:
        pass

    body = {
        "message": f"S25 signal: {content_dict.get('action','?')} {content_dict.get('symbol','?')} @ {content_dict.get('timestamp','')}",
        "content": content_b64,
    }
    if sha:
        body["sha"] = sha

    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=GH_HEADERS, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            code = r.getcode()
            print(f"  [GH] {GH_FILE} pushed ({code})")
            return code
    except urllib.error.HTTPError as e:
        print(f"  [GH] Push error {e.code}: {e.read().decode()[:200]}")
        return e.code


def build_gemini_prompt(signal_text):
    return (
        f"Tu es ARKON-5, IA trading crypto du pipeline S25 Lumiere.\n"
        f"Signal recu: {signal_text}\n\n"
        f"Analyse ce signal et reponds UNIQUEMENT en JSON valide (pas de markdown, pas de ```), format exact:\n"
        f'{{"action":"BUY|SELL|HOLD","confidence":0.XX,"pair":"XXX/USDT","take_profit":PRICE_OR_NULL,"stop_loss":PRICE_OR_NULL,"reason":"une phrase max"}}'
    )


def process_signal(prompt_text):
    print(f"\n[S25] Nouveau signal: {prompt_text[:80]}")
    print(f"  [1/3] Appel Gemini...")

    gemini_prompt = build_gemini_prompt(prompt_text)
    raw = ha_gemini(gemini_prompt)

    if not raw:
        print("  [!] Gemini no response - skip")
        ha_set("input_text.ai_response", "ARKON-5: ERREUR Gemini timeout")
        return

    print(f"  [2/3] Gemini raw: {raw[:120]}")

    sig = parse_gemini_signal(raw, prompt_text)
    conf_pct = int(sig["confidence"] * 100)
    action   = sig["action"]
    symbol   = sig["symbol"]
    reason   = sig["reason"]
    tp       = sig["take_profit"]
    sl       = sig["stop_loss"]

    tp_str = f" | TP:{tp}" if tp else ""
    sl_str = f" SL:{sl}" if sl else ""
    arkon_resp = f"ARKON-5: {action} {symbol} (conf: {conf_pct}%){tp_str}{sl_str} | {reason}"
    arkon_resp = arkon_resp[:255]

    code1 = ha_set("input_text.ai_response", arkon_resp)
    code2 = ha_set("input_text.ai_model_actif", "ARKON5_GEMINI_BRIDGE")
    print(f"  HA update: ai_response={code1}, ai_model_actif={code2}")

    print(f"  [3/3] Push GitHub...")
    push_github(sig)

    print(f"  DONE: {action} {symbol} conf={conf_pct}%")


def main():
    global last_prompt
    print("=" * 55)
    print("  S25 LUMIERE - Gemini Bridge  v1.0")
    print(f"  HA: {HA_URL}")
    print(f"  Gemini agent: {GEMINI_AGENT or 'non configure'}")
    print(f"  Poll: {POLL_SEC}s")
    print("=" * 55)

    init = ha_get("input_text.ai_prompt")
    if init:
        last_prompt = init.get("state", "")
        print(f"[INIT] prompt actuel: {last_prompt[:60] or '(vide)'}")
    else:
        print("[INIT] HA non accessible - verif URL et token dans s25_config.py")
        sys.exit(1)

    print("[RUN] En attente de signaux Kimi...\n")

    while True:
        try:
            data = ha_get("input_text.ai_prompt")
            if data:
                current = data.get("state", "")
                if current and current != last_prompt and current not in ("unknown", "unavailable", "none", ""):
                    last_prompt = current
                    process_signal(current)
                else:
                    print(".", end="", flush=True)
            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            print("\n[S25] Bridge arrete. Bye!")
            break
        except Exception as e:
            print(f"\n[ERR] {e}")
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
