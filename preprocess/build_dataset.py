import math

import numpy as np


def split_patients(patient_admission, admission_codes, code_map, train_num, test_num, seed=6669):
    np.random.seed(seed)
    common_pids = set()
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission["adm_id"]]
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break
    print('\r\t100%')
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    valid_num = len(patient_admission) - train_num - test_num
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids

def build_code_xy(pids: np.ndarray,
                  patient_admission: dict,
                  admission_codes_encoded: dict,
                  max_admission_num: int,
                  code_num: int,
                  max_code_num_in_a_visit: int) -> (np.ndarray, np.ndarray, np.ndarray):
    print('building train/valid/test codes features and labels ...')
    n = len(pids)
    x = np.zeros((n, max_admission_num, max_code_num_in_a_visit), dtype=int)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n, ), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission['admission_id']]
            x[i][k][:len(codes)] = codes
        codes = np.array(admission_codes_encoded[admissions[-1]['admission_id']]) - 1
        y[i][codes] = 1
        lens[i] = len(admissions) - 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens


def build_code_xy_with_subsets(pids: np.ndarray,
                               patient_admission: dict,
                               admission_codes_encoded: dict,
                               max_admission_num: int,
                               code_num: int,
                               max_code_num_in_a_visit: int,
                               heart_failure_indices, hypertensive_indices, cerebrovascular_indices,
                               sepsis_indices, acute_renal_failure_indices, diabetes_indices) -> (np.ndarray, np.ndarray, np.ndarray, dict):
    print('building train/valid/test codes features and labels ...')
    n = len(pids)
    x = np.zeros((n, max_admission_num, max_code_num_in_a_visit), dtype=int)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n,), dtype=int)
    heart_failure_mask = np.zeros((n,), dtype=bool)
    hypertensive_mask = np.zeros((n,), dtype=bool)
    cerebrovascular_mask = np.zeros((n,), dtype=bool)
    sepsis_mask = np.zeros((n,), dtype=bool)
    acute_renal_failure_mask = np.zeros((n,), dtype=bool)
    diabetes_mask = np.zeros((n,), dtype=bool)

    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission['adm_id']]
            x[i][k][:len(codes)] = codes
        codes = np.array(admission_codes_encoded[admissions[-1]['adm_id']]) - 1
        y[i][codes] = 1
        lens[i] = len(admissions) - 1

        heart_failure_mask[i] = np.any(y[i, heart_failure_indices] > 0)
        hypertensive_mask[i] = np.any(y[i, hypertensive_indices] > 0)
        cerebrovascular_mask[i] = np.any(y[i, cerebrovascular_indices] > 0)
        sepsis_mask[i] = np.any(y[i, sepsis_indices] > 0)
        acute_renal_failure_mask[i] = np.any(y[i, acute_renal_failure_indices] > 0)
        diabetes_mask[i] = np.any(y[i, diabetes_indices] > 0)

    print('\r\t%d / %d' % (len(pids), len(pids)))

    subset_masks = {
        'heart_failure': heart_failure_mask,
        'hypertensive': hypertensive_mask,
        'cerebrovascular': cerebrovascular_mask,
        'sepsis': sepsis_mask,
        'acute_renal_failure': acute_renal_failure_mask,
        'diabetes': diabetes_mask
    }
    return x, y, lens, subset_masks
