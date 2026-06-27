#!/usr/bin/env bash
# cora_server_lib.sh — shared client helpers for the CORA background server, sourced by
# prepare_instance.sh and run_instance.sh. Centralises the lease + log-relay + wait logic so the
# kill-safety details (close the lease fd in children; release on any exit) live in one place.
#
# Globals it reads:  CORA_SERVER_DIR, CORA_PING_TIMEOUT
# Globals it sets:   SRV_DIR, JOBID

SRV_DIR="${CORA_SERVER_DIR:-${HOME}/.cora_server}"
PING_TIMEOUT="${CORA_PING_TIMEOUT:-15}"

# Liveness: write `ping`, wait up to $1 seconds for the (idle) daemon to touch `pong`.
ping_ok() {
    rm -f "$SRV_DIR/pong"; : > "$SRV_DIR/ping"
    local n=$(( ${1:-$PING_TIMEOUT} * 10 ))
    while [ "$n" -gt 0 ]; do
        [ -e "$SRV_DIR/pong" ] && { rm -f "$SRV_DIR/pong"; return 0; }
        sleep 0.1; n=$((n - 1))
    done
    return 1
}

server_alive() {
    [ -f "$SRV_DIR/server.pid" ] && kill -0 "$(cat "$SRV_DIR/server.pid" 2>/dev/null)" 2>/dev/null
}

# submit_job TYPE BENCH ONNX VNNLIB TIMEOUT OVERRIDES WAIT_TOTAL
#   Publishes a job, relays its live log to stdout (-> run.sh tee -> website), and waits up to
#   WAIT_TOTAL seconds for the daemon's `done`. Holds an flock'd lease for the call's lifetime;
#   the kernel releases it on ANY exit (incl. untrappable SIGKILL) so cora_server.sh can detect a
#   killed owner. The daemon writes the verdict to $SRV_DIR/result (run) and a return code to
#   $SRV_DIR/prep_rc.
#   Returns: 0 done; 1 could not acquire lease; 2 timed out (caller should exit to free the lease
#   so the wedged daemon is torn down).
submit_job() {
    local type="$1" bench="$2" onnx="$3" vnnlib="$4" timeout="$5" overrides="$6" wait_total="$7"
    JOBID="$$-$(date +%s%N)"

    # Hold the lease for our whole lifetime. The kernel drops it on any exit (incl. SIGKILL).
    exec 9> "$SRV_DIR/lease"
    flock -w 10 9 || return 1

    # We own the channel: clear stale per-job files, then publish the request atomically. Pass
    # cwd so the daemon resolves the same relative onnx/vnnlib paths and the prepare->run .mat
    # handoff (a bare filename written to the CWD) lands consistently.
    rm -f "$SRV_DIR/done" "$SRV_DIR/result" "$SRV_DIR/running" "$SRV_DIR/prep_rc"
    : > "$SRV_DIR/job.log"
    { echo "id=$JOBID"; echo "type=$type"; echo "cwd=$(pwd)"; echo "bench=$bench";
      echo "onnx=$onnx"; echo "vnnlib=$vnnlib"; echo "timeout=$timeout";
      echo "overrides=$overrides"; } > "$SRV_DIR/request.tmp"
    mv "$SRV_DIR/request.tmp" "$SRV_DIR/request"

    # Relay the daemon's per-job log. CRITICAL: close the lease fd in this child (9>&-), else tail
    # inherits fd 9 and keeps the flock'd open-file-description alive after we are SIGKILLed, so
    # the kernel would NOT release the lease and cora_server.sh could never detect our death.
    tail -n +1 -F -s 0.2 "$SRV_DIR/job.log" 2>/dev/null 9>&- &
    local tail_pid=$!
    trap 'kill "$tail_pid" 2>/dev/null; exit 124' TERM INT

    local deadline=$(( $(date +%s) + wait_total ))
    while :; do
        [ "$(cat "$SRV_DIR/done" 2>/dev/null || true)" = "$JOBID" ] && break
        if [ "$(date +%s)" -ge "$deadline" ]; then
            kill "$tail_pid" 2>/dev/null; trap - TERM INT
            return 2   # caller exits -> lease frees -> cora_server.sh kills the wedged daemon
        fi
        sleep 0.2 9>&-   # 9>&-: don't let the sleep child hold the lease fd during a kill
    done

    sleep 0.5            # let tail drain the final lines before we stop it
    kill "$tail_pid" 2>/dev/null; trap - TERM INT
    return 0
}
