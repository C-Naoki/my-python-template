#!/bin/sh

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd) || exit 1

USE_NOHUP=0

sanitize_filename() {
  printf '%s' "$1" | sed 's#[/[:space:]]#_#g; s#[^A-Za-z0-9._=-]#_#g'
}

if [ "${1:-}" = "-n" ]; then
  USE_NOHUP=1
  shift
fi

if [ "$#" -eq 0 ]; then
  echo "Error: COMMAND is required" >&2
  echo "Usage: $0 [-n] command" >&2
  exit 1
fi

COMMAND="$*"

case "$COMMAND" in
"uv run" | "uv run "*) ;;
*)
  echo "Error: Invalid options" >&2
  echo "Use only: [-n]" >&2
  exit 1
  ;;
esac

cd "$ROOT_DIR" || exit 1

sh bin/add_pth.sh || exit 1

MODEL_NAME=$(printf '%s\n' "$COMMAND" | sed -n 's/.*model=\([^[:space:]]*\).*/\1/p')
MODEL_NAME=$(sanitize_filename "${MODEL_NAME:-unknown_model}")

INPUT_DIR=$(printf '%s\n' "$COMMAND" | sed -n 's/.*io.input_dir=\([^[:space:]]*\).*/\1/p')
INPUT_DIR=$(sanitize_filename "${INPUT_DIR:-unknown_input}")

if [ "$USE_NOHUP" -eq 1 ]; then
  DATE_DIR=$(date +%Y%m%d)
  TARGET_DIR="nohup/$DATE_DIR"
  mkdir -p "$TARGET_DIR"

  OUTFILE="${TARGET_DIR}/${MODEL_NAME}_${INPUT_DIR}.out"
  if [ -e "$OUTFILE" ]; then
    base="${OUTFILE%.out}"
    ext=".out"
    i=2
    while [ -e "${base}_${i}${ext}" ]; do
      i=$((i + 1))
    done
    OUTFILE="${base}_${i}${ext}"
  fi
  nohup sh -c "$COMMAND" > "$OUTFILE" 2>&1 &
else
  sh -c "$COMMAND"
fi
