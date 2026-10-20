#!/usr/bin/env bash
set -euo pipefail
VOICE_LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$VOICE_LAB_ROOT/.venv-run/bin/python" ]]; then
    echo '未找到项目运行环境。请按 README.md 创建 .venv-run。' >&2
    exit 1
fi
exec "$VOICE_LAB_ROOT/.venv-run/bin/python" "$VOICE_LAB_ROOT/main.py" "$@"
