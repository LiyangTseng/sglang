#!/bin/bash
# Runs every measurement in `measure.rs` for one method, from `rust/`:
#   sglang-server/src/api_server/http/request_decompression/measure_arms/run.sh <method>
# <method>: read-then-permit (original), permit-then-read, byte-budget (the PR).
# The first two copy that method's `decompress_zstd` onto the PR's `Limits`
# plumbing, so one harness drives all three; the file is restored on exit.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
target=$here/../../request_decompression.rs
case "${1:-}" in
  read-then-permit) cp "$here/read_then_permit.rs" "$target" ;;
  permit-then-read) cp "$here/permit_then_read.rs" "$target" ;;
  byte-budget) ;;
  *) echo "usage: $0 read-then-permit|permit-then-read|byte-budget" >&2; exit 2 ;;
esac
trap 'git checkout -- "$target"' EXIT
M="cargo test --release -q -p sglang-server --lib request_decompression::measure::"
O="-- --ignored --nocapture --test-threads=1"
for run in 1 2 3; do
  for known in 1 0; do
    for n in 64 128; do
      ITEM9_N=$n ITEM9_KNOWN=$known ITEM9_BUDGET_MIB=1024 ${M}measure_concurrent_queued_memory $O
    done
  done
  for k in 0 4 8; do
    ITEM9_SLOW=$k ITEM9_BUDGET_MIB=1024 ${M}measure_slow_uploads $O
  done
done
ITEM9_P=4 ${M}measure_active_job_memory $O
${M}measure_decode_allocations $O
