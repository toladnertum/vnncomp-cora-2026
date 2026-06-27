#!/usr/bin/env bash
# run_instance.sh — VNN-COMP per-instance entry; a thin client to the CORA background server. It
# submits the instance as a `run` job, relays the daemon's live log to its own stdout (-> run.sh
# tee -> website), waits for the verdict, and copies it to the framework's out file. The daemon
# (run_instance.m) enforces the per-instance timeout itself and writes the verdict.
#
# It holds an flock'd lease for its whole lifetime; if it is killed by the framework's outer
# `timeout` (SIGTERM then SIGKILL) or the per-benchmark killer, the kernel drops the lease and
# cora_server.sh tears the daemon down for a clean restart — even though SIGKILL cannot run a
# trap. If no healthy server is available it falls back to a direct MATLAB run, so one bad server
# never loses a whole category.
#   args (vnncomp contract): v1 category onnx vnnlib out_file timeout
#
# The benchmark harness (run_single_instance.sh) runs with `set -x` and exports SHELLOPTS, so
# xtrace would otherwise propagate into this client and its 0.2s wait loop, drowning the relayed
# MATLAB job output in hundreds of `+ sleep 0.2` / `++ cat .../done` trace lines. Turn it off
# first; the actual verification output is the diary relay (tail -F job.log), unaffected by this.
set +x
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/cora_server_lib.sh"

CATEGORY="$2"; ONNX="$3"; VNNLIB="$4"; OUT="$5"; TIMEOUT="$6"
WAIT_GRACE="${CORA_RUN_GRACE:-30}"     # extra wait beyond TIMEOUT for the daemon's result

# Direct fallback: a plain MATLAB run, matching CORA's non-server run_instance.sh.
# CORA_DIRECT_CMD is a test seam: `<cmd> run <bench> <onnx> <vnnlib> <out> <timeout>`.
direct_run() {
    echo "[run_instance] no healthy server; running directly via MATLAB"
    if [ -n "${CORA_DIRECT_CMD:-}" ]; then
        $CORA_DIRECT_CMD run "$CATEGORY" "$ONNX" "$VNNLIB" "$OUT" "$TIMEOUT"; return $?
    fi
    local b="${CATEGORY//\'/\'\'}" o="${ONNX//\'/\'\'}" v="${VNNLIB//\'/\'\'}" r="${OUT//\'/\'\'}"
    sudo matlab -batch "addpath(genpath('$HERE')); run_instance('$b','$o','$v','$r',$TIMEOUT,true);"
}

# No live + responsive server -> fall back (don't lose the instance).
if ! server_alive || ! ping_ok; then
    direct_run
    exit 0
fi

submit_job "run" "$CATEGORY" "$ONNX" "$VNNLIB" "$TIMEOUT" "" "$((TIMEOUT + WAIT_GRACE))"
case $? in
    0)  cp "$SRV_DIR/result" "$OUT"
        exit 0 ;;
    1)  echo "[run_instance] could not acquire server lease; running directly"
        direct_run; exit 0 ;;
    2)  echo "[run_instance] no result within $((TIMEOUT + WAIT_GRACE))s; aborting (server will restart)"
        exit 124 ;;   # exiting frees the lease -> cora_server.sh kills the wedged daemon
esac
