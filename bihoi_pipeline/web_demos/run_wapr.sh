#!/usr/bin/env bash
set -euo pipefail
DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$DEMO_DIR/../scripts/common.sh"
exec "${WEB_PYTHON:-python3}" "$DEMO_DIR/server.py" --app wapr --port 7863 --data-dir "$DEMO_DIR/runtime/wapr" "$@"
