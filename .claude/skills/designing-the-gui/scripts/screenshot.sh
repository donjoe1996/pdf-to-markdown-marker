#!/usr/bin/env bash
# Screenshot every tab of app.py against a demo library, without touching the
# real one. Usage:  bash .claude/skills/designing-the-gui/scripts/screenshot.sh NAME
# Writes $OUT/NAME-<tab>.png (OUT defaults to a temp dir; printed at the end).
#
# Works on a *copy* of the app: bt.queue discovers PDFs relative to its own
# file, so the copy sees only the seeded demo document, and a running worker on
# the real library is never disturbed.
set -euo pipefail
NAME=${1:?usage: screenshot.sh NAME}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-${TMPDIR:-/tmp}/gui-shots}
PORT=${PORT:-8599}
WORK=$(mktemp -d)
mkdir -p "$OUT"
cp -r "$REPO/app.py" "$REPO/bt" "$REPO/.streamlit" "$WORK/"
"$REPO/.venv/bin/python" "$HERE/seed_demo.py" "$WORK"
( cd "$WORK" && exec "$REPO/.venv/bin/streamlit" run app.py \
    --server.headless true --server.port "$PORT" >"$WORK/streamlit.log" 2>&1 ) &
PID=$!
trap 'kill $PID 2>/dev/null; rm -rf "$WORK"' EXIT
sleep 8
node "$HERE/shoot.js" "http://localhost:$PORT" "$OUT/$NAME"
echo "screenshots in $OUT"
