#!/bin/bash
# Sync this submission repo from the CORA repo's vnncomp2026-main branch.
#
# code/cora/           <- git archive of the branch (tracked files only)
# <root .m/.sh files>  <- examples/nn/vnncomp/ from the same snapshot
#
# config.yaml, Dockerfile, README.md and this script are submission-only and
# stay untouched. Commits only if the sync changed something; pushing is a
# deliberate manual step.
set -euo pipefail

CORA=/home/benedikt/Documents/MATLAB/Toolboxes/cora
BRANCH=vnncomp2026-main
SELF="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

HASH=$(git -C "$CORA" rev-parse --short "$BRANCH")

# extract a clean snapshot of the branch
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git -C "$CORA" archive "$BRANCH" | tar -x -C "$TMP"

# full CORA toolbox under code/cora
rm -rf "$SELF/code/cora"
mkdir -p "$SELF/code/cora"
cp -a "$TMP"/. "$SELF/code/cora/"

# platform entry points at the repo root
for f in prepare_instance.m run_instance.m getInstanceFilename.m \
         printErrorMessage.m writeCounterexample.m \
         prepare_instance.sh run_instance.sh install_tool.sh post_install.sh; do
    cp "$TMP/examples/nn/vnncomp/$f" "$SELF/$f"
done

cd "$SELF"
git add -A
if git diff --cached --quiet; then
    echo "Already in sync with $BRANCH @ $HASH"
    exit 0
fi
git commit -m "Sync with cora $BRANCH @ $HASH"
echo "Synced: $BRANCH @ $HASH (committed locally; push when ready: git -C $SELF push)"
