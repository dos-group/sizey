#!/usr/bin/env bash
set -euo pipefail

# Config
DATA_DIR="./data"
FILES=(
  "trace_chipseq.csv"
  "trace_eager.csv"
  "trace_iwd.csv"
  "trace_mag.csv"
  "trace_methylseq.csv"
  "trace_rnaseq.csv"
)
ALPHAS=("0.0" "0.25" "0.5" "0.75" "1.0")
SOFTMAX=("True" "False")
SEEDS=("1001" "1111" "1234" "1996" "2024")   # five fixed 4-digit seeds
ERROR_METRIC="smoothed_mape"

LOG_DIR="sweep_logs"
mkdir -p "$LOG_DIR"

run_one () {
  local file="$1"
  local alpha="$2"
  local softmax="$3"
  local seed="$4"
  local use_grid_flag="$5"   # "" or "--use_online_grid"

  # Create unique log filename
  local base_name
  base_name=$(basename "$file" .csv)
  local grid_tag
  if [ -z "$use_grid_flag" ]; then
    grid_tag="nogrid"
  else
    grid_tag="grid"
  fi
  local log_file="${LOG_DIR}/${base_name}_a${alpha}_s${softmax}_seed${seed}_${grid_tag}.log"

  echo ">>> Running: ${file}, alpha=${alpha}, softmax=${softmax}, seed=${seed}, ${grid_tag}"
  /usr/bin/time -f "Run time: %E (elapsed), %U (user), %S (sys)" \
    python main.py "${DATA_DIR}/${file}" "${alpha}" "${softmax}" "${ERROR_METRIC}" "${seed}" ${use_grid_flag} \
    >"$log_file" 2>&1
}

# Sweep
for file in "${FILES[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    for soft in "${SOFTMAX[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_one "$file" "$alpha" "$soft" "$seed" ""
        run_one "$file" "$alpha" "$soft" "$seed" "--use_online_grid"
      done
    done
  done
done

echo "All runs completed. Logs saved in $LOG_DIR/"
