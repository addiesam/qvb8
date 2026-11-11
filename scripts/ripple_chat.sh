#!/usr/bin/env bash
# Terminal chat with a running Ripple-65B server (API client only: no model is loaded here).
#   scripts/ripple_chat.sh                                   interactive, server on port 8080
#   PORT=8080 scripts/ripple_chat.sh "Explain entropy" --no-interactive
#   URL=http://host:8080 scripts/ripple_chat.sh --no-think
# Env: PORT (default 8080), URL (overrides PORT), PYTHON (default: python). Other flags: python -m ai_engine.cli.chat_client --help
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
exec "${PYTHON:-python}" -m ai_engine.cli.chat_client --url "${URL:-http://127.0.0.1:${PORT:-8080}}" "$@"
