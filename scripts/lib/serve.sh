# Shared start / stop / status for the launchers (scripts/*.sh). Source it, do not run it.
# The launcher sets:
#   NAME      log / pid name, e.g. Ripple-65B_1e-3
#   DEF_PORT  HTTP port (env PORT overrides)
#   launch()  runs the server in the FOREGROUND (exec ...); sees PORT and EXTRA (all non-action arguments)
# then calls:  serve_main "$@"
#
#   <script> [start|stop|status|restart|log|startup] [extra server flags...]      (start is the default)
# Log: scripts/logs/<NAME>.log  (pid: <NAME>.pid).
# Startup log: scripts/logs/<NAME>.startup.log -- the command line and everything the server printed until it
# answered /health (or until it died): the load / placement / kernel / memory lines, kept apart from the request log.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

_g_pid() {      # the server's pid: pid file first, else whatever listens on the port
  local p
  p=$(cat "$PIDF" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo "$p"; return; }
  ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$" {print $NF}' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}

_g_stop() {
  local pid; pid=$(_g_pid)
  [ -z "$pid" ] && { echo "[$NAME] not running"; rm -f "$PIDF"; return 0; }
  kill -- -"$pid" 2>/dev/null || kill "$pid"            # the whole process group when started here (setsid)
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$pid" 2>/dev/null && { kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid"; sleep 1; }
  rm -f "$PIDF"; echo "[$NAME] stopped (pid $pid)"
}

_g_status() {
  local pid; pid=$(_g_pid)
  if [ -n "$pid" ]; then echo "[$NAME] running, pid $pid, port $PORT, log $LOGF"
  else echo "[$NAME] stopped (log $LOGF)"; fi
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
}

_g_start() {
  local pid; pid=$(_g_pid)
  [ -n "$pid" ] && { echo "[$NAME] already running (pid $pid, port $PORT) -- $0 stop first"; return 0; }
  mkdir -p "$LOGDIR"
  echo "[$NAME] starting on port $PORT  (log: $LOGF, startup log: $STARTF)"
  QV_SERVE_RUN=1 PORT="$PORT" setsid nohup "$SELF" start "${EXTRA[@]}" > "$LOGF" 2>&1 < /dev/null &
  pid=$!; echo "$pid" > "$PIDF"
  local t0; t0=$(date +%s)
  _g_startup_log() {          # $1 = outcome: the command, then the server output so far
    { echo "# $(date '+%F %T')  $SELF start ${EXTRA[*]}  (port $PORT, pid $pid) -- $1 after $(( $(date +%s) - t0 ))s"
      echo "# env: MODEL=${MODEL:-} DRAFT=${DRAFT:-}"
      cat "$LOGF"; } > "$STARTF"
  }
  for _ in $(seq 360); do                               # up to 30 min (a 65B pack loads in a few minutes)
    curl -sf -o /dev/null -m 3 "http://127.0.0.1:$PORT/health" && {
      _g_startup_log ready
      echo "[$NAME] ready: http://$(hostname -I | awk '{print $1}'):$PORT/  (pid $pid, startup log $STARTF)"; return 0; }
    kill -0 "$pid" 2>/dev/null || { _g_startup_log FAILED; echo "[$NAME] FAILED -- last lines of $LOGF:"; tail -5 "$LOGF"; rm -f "$PIDF"; exit 1; }
    sleep 5
  done
  _g_startup_log "still starting after 30 min"
  echo "[$NAME] still starting after 30 min, see $LOGF"
}

serve_main() {
  local action=start a
  EXTRA=()
  for a in "$@"; do
    case "$a" in
      start|stop|status|restart|log|startup) action="$a" ;;
      -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
      *) EXTRA+=("$a") ;;
    esac
  done
  PORT="${PORT:-$DEF_PORT}"
  LOGDIR="$ROOT/scripts/logs"; LOGF="$LOGDIR/$NAME.log"; PIDF="$LOGDIR/$NAME.pid"; STARTF="$LOGDIR/$NAME.startup.log"
  cd "$ROOT" || exit 1
  if [ -n "${QV_SERVE_RUN:-}" ]; then launch; exit $?; fi       # the detached child: run the server itself
  case "$action" in
    start)   _g_start ;;
    stop)    _g_stop ;;
    status)  _g_status ;;
    restart) _g_stop; _g_start ;;
    log)     exec tail -f "$LOGF" ;;
    startup) exec cat "$STARTF" ;;
  esac
}
