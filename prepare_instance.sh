#!/usr/bin/env bash
# prepare_instance.sh — VNN-COMP per-instance prep. Ensures the CORA background server is up
# (lazily starting cora_server.sh on the first instance, restarting it if a previous run left it
# dead/wedged), then submits the parse/prepare as a job to the warm daemon. MATLAB+CORA startup
# is therefore paid once per server lifetime, not per instance. Output here is relayed to the
# website. Exits with prepare_instance.m's return code (nonzero -> the framework skips the
# category, per the rules); falls back to a direct MATLAB run if no server can be brought up.
#   args (vnncomp contract): v1 category onnx vnnlib
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/cora_server_lib.sh"

CATEGORY="$2"; ONNX="$3"; VNNLIB="$4"
SERVER_SH="${CORA_SERVER_SH:-$HERE/cora_server.sh}"
START_TIMEOUT="${CORA_START_TIMEOUT:-550}"     # server boot budget; under the 600s prepare cap
PREP_WAIT="${CORA_PREP_WAIT:-590}"             # wait for the prepare job; under the 600s cap
OVERRIDES="${CORA_OVERRIDES_FILE:-}"
mkdir -p "$SRV_DIR"

# Direct fallback: a plain MATLAB run, matching CORA's non-server prepare_instance.sh.
# CORA_DIRECT_CMD is a test seam: `<cmd> prepare <bench> <onnx> <vnnlib>`.
direct_prepare() {
    echo "[prepare] no healthy server; preparing directly via MATLAB"
    if [ -n "${CORA_DIRECT_CMD:-}" ]; then
        $CORA_DIRECT_CMD prepare "$CATEGORY" "$ONNX" "$VNNLIB"; return $?
    fi
    local b="${CATEGORY//\'/\'\'}" o="${ONNX//\'/\'\'}" v="${VNNLIB//\'/\'\'}"
    if [ -n "$OVERRIDES" ] && [ -f "$OVERRIDES" ]; then
        local ov="${OVERRIDES//\'/\'\'}"
        sudo matlab -batch "addpath(genpath('$HERE')); r=prepare_instance('$b','$o','$v','$ov'); exit(double(r));"
    else
        sudo matlab -batch "addpath(genpath('$HERE')); r=prepare_instance('$b','$o','$v'); exit(double(r));"
    fi
}

ensure_server() {
    if server_alive && ping_ok "$PING_TIMEOUT"; then
        echo "[prepare] CORA server already running and responsive"
        return 0
    fi
    echo "[prepare] starting CORA background server..."
    if [ -f "$SRV_DIR/server.pid" ]; then
        kill -TERM "$(cat "$SRV_DIR/server.pid" 2>/dev/null)" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$SRV_DIR/running" "$SRV_DIR/done" "$SRV_DIR/result" "$SRV_DIR/request" "$SRV_DIR/prep_rc" "$SRV_DIR/pong"
    # Detached, own session, so the per-benchmark killer can't reap it with the measurement tree.
    setsid bash "$SERVER_SH" >>"$SRV_DIR/server.log" 2>&1 < /dev/null &
    # Relay the boot log (MATLAB starting) to the website while we wait for readiness.
    : >> "$SRV_DIR/server.log"
    tail -n 0 -F -s 0.2 "$SRV_DIR/server.log" 2>/dev/null & local boot_tail=$!
    local n="$START_TIMEOUT"
    while [ "$n" -gt 0 ]; do
        if server_alive && ping_ok 1; then
            echo "[prepare] CORA server is up and responsive"
            kill "$boot_tail" 2>/dev/null || true
            return 0
        fi
        sleep 1; n=$((n - 1))
    done
    kill "$boot_tail" 2>/dev/null || true
    return 1
}

if ! ensure_server; then
    echo "[prepare] CORA server failed to start; falling back to a direct run"
    direct_prepare; exit $?
fi

# Submit the prepare job to the warm daemon.
submit_job "prepare" "$CATEGORY" "$ONNX" "$VNNLIB" "0" "$OVERRIDES" "$PREP_WAIT"
case $? in
    0)  rc="$(cat "$SRV_DIR/prep_rc" 2>/dev/null || echo 0)"
        echo "[prepare] done (return code $rc)"
        exit "$rc" ;;
    1)  echo "[prepare] could not acquire server lease; preparing directly"
        direct_prepare; exit $? ;;
    2)  echo "[prepare] prepare job exceeded ${PREP_WAIT}s; skipping category (server will restart)"
        exit 1 ;;   # nonzero -> category skipped; exiting frees the lease -> daemon torn down
esac
