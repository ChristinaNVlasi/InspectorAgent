#!/usr/bin/env bash
# Stop all SMA platform processes
LOGS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"

echo "⛔  Stopping SMA Inspector Platform …"

# Kill by saved PID files first
for pid_file in global_mcp knorr_agent arcelik_agent borg_agent noise_api guidance_agent; do
  f="$LOGS/${pid_file}.pid"
  if [[ -f "$f" ]]; then
    pid=$(cat "$f")
    kill "$pid" 2>/dev/null && echo "   stopped $pid_file (pid $pid)" || true
    rm -f "$f"
  fi
done

sleep 1

# Belt-and-braces: force-kill anything still on these ports
for port in 3030 2829 2828 2827 5001 2830; do
  pids=$(lsof -ti :"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "   force-killing port $port: $pids"
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
done

sleep 1
echo "✔  All services stopped."
