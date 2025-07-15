#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Usage: ./rerun_failed_tasks.sh <ENSEMBLE_RUN>
# Example: ./rerun_failed_tasks.sh 05
# ----------------------------------------

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <ENSEMBLE_RUN>"
  exit 1
fi

ENSEMBLE_RUN=$(($1))

eval "$(conda shell.bash hook)"
export ENSEMBLE_RUN
#---------------------------------------------------------------
# 1) Adjust these paths/vars to match your directory structure
# ----------------------------------------------------------------
#FAIR_VERSION="2.1.3"
#CALIBRATION_VERSION="1.4"
#CONSTRAINT_SET="all-2022"
if [ -f .env1 ]; then
  export $(cat .env1 | xargs)
fi


# If your "cut‐year" starts at 1930 and ends at 2024, adjust here:
CUTOFF_MIN=1930
CUTOFF_MAX=2099
echo $CONSTRAINT_SET
# `script_dir` is assumed to be the parent of "input/" and "output/"
script_dir="$(cd "$(dirname "$0")" && pwd)"

# Where the *input* Python scripts live:
CONSTRAIN_DIR="${script_dir}/input/fair-${FAIR_VERSION}/v${CALIBRATION_VERSION}/${CONSTRAINT_SET}/constraining"

# Base of your *output* folder tree:
OUTPUT_BASE="${script_dir}/output/fair-${FAIR_VERSION}/v${CALIBRATION_VERSION}/${CONSTRAINT_SET}"
conda activate fair-calibrate
# ----------------------------------------------------------------
# 2) Helper functions
# ----------------------------------------------------------------

# under the fair-calibrate environment for a single cutoff year.
process_cutoff_o(){
  local cutoff=$1
  local script_name=$2
  export CUTOFF_YEAR=$cutoff
  # The output subfolder for this ENSEMBLE_RUN and CUTOFF_YEAR:
  cd ${CONSTRAIN_DIR}
  # Run the requested Python script (must exist in $CONSTRAIN_DIR)
  python "${script_name}"
}

# process_cutoff_n: run 04_dump-calibration.py under the fair2 environment
process_cutoff_n(){
  local cutoff=$1
  export CUTOFF_YEAR=$cutoff
  cd ${CONSTRAIN_DIR}
  # Run the dump‐calibration script (assumed to live in CALIBRATION_DIR)
  python "04_dump-calibration.py"
}


# ----------------------------------------------------------------
# 3) Loop over cutoff years: check‐and‐rerun 01_… and 03_… 
# ----------------------------------------------------------------
echo "===== STARTING: rerun 01 + 03 for any missing outputs ====="
for cutoff in $(seq "$CUTOFF_MIN" "$CUTOFF_MAX"); do
  echo -n "cut=$cutoff"
  outdir="${OUTPUT_BASE}r${ENSEMBLE_RUN}cut${cutoff}/posteriors"
  #echo $outdir
  # 3.1 Check runids_rmse_pass.csv → 01_constrain-gsat-rmse-only.py
  file1="runids_rmse_pass.csv"
  if [[ ! -s "${outdir}/${file1}" ]]; then
    echo "Missing or empty ${file1}; rerunning 01_constrain-gmst-rmse-only.py for cutoff ${cutoff}"
    process_cutoff_o "$cutoff" "01_constrain-gsat-rmse-only.py"
  else
    echo -n "✓"
  fi

  # 3.2 Check runids_rmse_reweighted_pass.csv → 03_reweight-rmse-posterior-multiple-constraints.py
  file2="runids_rmse_reweighted_pass.csv"
  if [[ ! -s "${outdir}/${file2}" ]]; then
    echo "Missing or empty ${file2}; rerunning 03_reweight-rmse-posterior-multiple-constraints.py for cutoff ${cutoff}"
    process_cutoff_o "$cutoff" "03_reweight-rmse-posterior-multiple-constraints.py"
  else
    echo -n "✓"
  fi
done
echo "===== DONE: reran 01 + 03 where needed ====="


# ----------------------------------------------------------------
# 4) Loop over cutoff years: check‐and‐rerun 04_dump-calibration.py
# ----------------------------------------------------------------
echo
echo "===== STARTING: rerun 04 for any missing outputs ====="
conda activate fair2
for cutoff in $(seq "$CUTOFF_MIN" "$CUTOFF_MAX"); do
  echo -n "cut=$cutoff"
  outdir="${OUTPUT_BASE}r${ENSEMBLE_RUN}cut${cutoff}/posteriors"
  out2dir="${script_dir}/output/${CONSTRAINT_SET}"
  # Check calibrated_constrained_parameters.csv → 04_dump-calibration.py
  file3="calibrated_constrained_parameters.csv"
  if [[ ! -s "${outdir}/${file3}" ]] || [[ ! -s "${out2dir}/${CONSTRAINT_SET}r${ENSEMBLE_RUN}cut${cutoff}_temp_all.npy" ]]; then  
    echo "Missing or empty ${file3}; rerunning 04_dump-calibration.py for cutoff ${cutoff}"
    process_cutoff_n "$cutoff"
  else
    echo -n "✓"
  fi
done
echo "===== DONE: reran 04 where needed ====="

echo
echo "All requested reruns complete."

