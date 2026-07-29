#!/usr/bin/env bash
# Send a test message to the configured Telegram chat.
# Usage: scripts/test-telegram.sh [chat_id]
set -uo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

CHAT="${1:-${TELEGRAM_CHAT_ID:-}}"

if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
  echo "TELEGRAM_BOT_TOKEN is not set (.env or environment)" >&2
  exit 1
fi

API="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}"

# Print the interesting fields, or Telegram's error, and set the exit code.
show() {
  python3 -c '
import json, sys
raw = sys.stdin.read().strip()
try:
    r = json.loads(raw)
except ValueError:
    print("  no JSON from Telegram (network/proxy issue?): " + raw[:200]); sys.exit(1)
if not r.get("ok"):
    print("  FAILED %s: %s" % (r.get("error_code"), r.get("description"))); sys.exit(1)
res = r["result"]
print("  ok: " + ", ".join("%s=%s" % (k, res[k]) for k in ("username", "message_id", "date") if k in res))
'
}

echo "== bot identity =="
curl -s "$API/getMe" | show || exit 1

if [ -z "$CHAT" ]; then
  echo
  echo "TELEGRAM_CHAT_ID is empty. Message the bot, then use one of these:" >&2
  curl -s "$API/getUpdates" | python3 -c '
import json, sys
for u in json.load(sys.stdin).get("result", []):
    c = (u.get("message") or u.get("channel_post") or {}).get("chat", {})
    if c:
        print("  chat_id=%s  type=%s  name=%s" % (c.get("id"), c.get("type"), c.get("title") or c.get("username")))
' | sort -u
  exit 1
fi

echo
echo "== sending to $CHAT =="
BODY=$(python3 -c '
import json, sys, datetime
print(json.dumps({"chat_id": sys.argv[1], "text":
  f"Test notifikasi Agnes\nWaktu: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n\nUser: halo\nAgnes: halo, ada yang bisa saya bantu?"}))
' "$CHAT")

curl -s -X POST "$API/sendMessage" -H 'Content-Type: application/json' -d "$BODY" | show || exit 1

echo
echo "OK - check your Telegram."
