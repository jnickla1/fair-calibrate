import numpy as np
import os
import re

def extract_rows_from_npy(ensemble_prefix, input_dir, output_file,tag="all", start_year=1930, end_year=2099, ncols=841, nrows=(2102-1750)):
    all_rows = []
    
    for year in range(start_year, end_year + 1):
        filename_pattern = f"{ensemble_prefix}cut{year}_temp_{tag}.npy"
        filepath = os.path.join(input_dir, filename_pattern)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filename_pattern} not found, skipping.")
            continue

        arr = np.load(filepath)

        # Pad to (275, 841) if needed
        if arr.shape[1] < ncols:
            padded = np.full((nrows, ncols), np.nan)
            padded[:, :arr.shape[1]] = arr
            arr = padded
        elif arr.shape[1] > ncols or arr.shape[0] != nrows:
            raise ValueError(f"{filename_pattern} has unexpected shape {arr.shape}, expected {nrows}")

        # Extract the row for this year
        row_index = year - 1750
        row = arr[row_index, :]
        all_rows.append(row)
    
    result = np.vstack(all_rows)  # shape (number of years, 841)
    np.save(output_file, result)
    print(f"Saved combined array with shape {result.shape} to {output_file}")

ensemble_name = 'MPIESM370'
for run in range(42,51):
    ens_pref= 'all_current_'+ensemble_name+'r'+str(run)
    for desig in ["all","nonat","anthro"]:
        extract_rows_from_npy(
            ensemble_prefix = ens_pref,
            input_dir='/users/jnickla1/data/jnickla1/fair-calibrate/output/all_current_'+ensemble_name,
            output_file='resliced/combined_'+ens_pref+'_'+desig+'.npy',
            tag=desig
        )

