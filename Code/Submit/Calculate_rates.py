import pandas as pd
import numpy as np
import pickle

VALUE_DATA_PKL_BASELINE = "./value_data_baseline.pkl" 
VALUE_DATA_PKL_AUGMENTED = "./value_data_augmented.pkl" 

#Red Flag Thresholds 
DN_RED_THRESHOLD = -0.25
RN_RED_THRESHOLD = 0.75

TRAJ_COL = 'traj'
CATEGORY_COL = 'category'
Q_DN_COL = 'q_dn' 
Q_RN_COL = 'q_rn'

def calculate_flag_rates(value_pkl_path):
    try:
        with open(value_pkl_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

    if not isinstance(data, pd.DataFrame):
         return None

    #Check required columns exist
    required_cols = [TRAJ_COL, CATEGORY_COL, Q_DN_COL, Q_RN_COL]
    if not all(col in data.columns for col in required_cols):
        return None

    flagged_survivors = 0
    total_survivors = 0
    flagged_nonsurvivors = 0
    total_nonsurvivors = 0

    # group by trajectory
    trajectory_groups = data.groupby(TRAJ_COL)
    unique_trajs = data[TRAJ_COL].unique()

    for traj_id in unique_trajs:
        group = trajectory_groups.get_group(traj_id)
        true_category = group[CATEGORY_COL].iloc[0]

        # median Q-values for each step in the trajectory
        try:
            v_dn_traj = np.array([np.median(q) for q in group[Q_DN_COL]], dtype=np.float32)
            v_rn_traj = np.array([np.median(q) for q in group[Q_RN_COL]], dtype=np.float32)
        except Exception as e:
            continue

        # See if any step in the trajectory meets red flag condition
        is_red_flagged_trajectory = np.any(
            (v_dn_traj <= DN_RED_THRESHOLD) & (v_rn_traj <= RN_RED_THRESHOLD)
        )

        #Update counts
        if true_category == 1:
            total_survivors += 1
            if is_red_flagged_trajectory:
                flagged_survivors += 1
        elif true_category == -1:
            total_nonsurvivors += 1
            if is_red_flagged_trajectory:
                flagged_nonsurvivors += 1

    fpr = (flagged_survivors / total_survivors) if total_survivors > 0 else 0
    tpr = (flagged_nonsurvivors / total_nonsurvivors) if total_nonsurvivors > 0 else 0

    return tpr, fpr

if __name__ == "__main__":
    tpr_base, fpr_base = calculate_flag_rates(VALUE_DATA_PKL_BASELINE)
    tpr_aug, fpr_aug = calculate_flag_rates(VALUE_DATA_PKL_AUGMENTED)

    print("Comparison:")
    if tpr_base is not None and fpr_base is not None:
        print(f"Baseline: TPR = {tpr_base:.4f}, FPR = {fpr_base:.4f}")
    else:
        print("Baseline: Calculation failed.")

    if tpr_aug is not None and fpr_aug is not None:
        print(f"Augmented: TPR = {tpr_aug:.4f}, FPR = {fpr_aug:.4f}")
    else:
        print("Augmented: Calculation failed.")