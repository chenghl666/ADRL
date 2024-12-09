import os
import _pickle as pickle

import numpy as np

from preprocess.parse_csv import parse_admission, parse_diagnoses
from preprocess.parse_csv import calibrate_patient_by_admission
from preprocess.encode import encode_code, encode_time_duration
from preprocess.build_dataset import split_patients
from preprocess.build_dataset import build_code_xy
from preprocess.auxiliary import co_occur


if __name__ == '__main__':
    data_path = 'data'
    raw_path = os.path.join(data_path, 'mimic3', 'raw')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/mimic3/raw`')
        exit()

    file_patient_admission = open(os.path.join(raw_path, 'patient_admission.pkl'), 'rb')
    patient_admission = pickle.load(file_patient_admission)

    file_admission_codes = open(os.path.join(raw_path, 'admission_codes.pkl'), 'rb')
    admission_codes = pickle.load(file_admission_codes)

    calibrate_patient_by_admission(patient_admission, admission_codes)


    print('There are %d valid patients' % len(patient_admission))

    max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
    max_code_num_in_a_visit = 0
    for admission_id, codes in admission_codes.items():
        if len(codes) > max_code_num_in_a_visit:
            max_code_num_in_a_visit = len(codes)

    admission_codes_encoded, code_map = encode_code(admission_codes)
    patient_time_duration_encoded = encode_time_duration(patient_admission)

    code_num = len(code_map)

    train_pids, valid_pids, test_pids = split_patients(
        patient_admission=patient_admission,
        admission_codes=admission_codes,
        code_map=code_map,
        train_num=6000,
        test_num=1000
    )


    mimic3_path = os.path.join('data', 'mimic3')
    encoded_path = os.path.join(mimic3_path, 'encoded')


    train_codes_x, train_codes_y, train_visit_lens = build_code_xy(train_pids, patient_admission, admission_codes_encoded, max_admission_num, code_num, max_code_num_in_a_visit)
    valid_codes_x, valid_codes_y, valid_visit_lens = build_code_xy(valid_pids, patient_admission, admission_codes_encoded, max_admission_num, code_num, max_code_num_in_a_visit)
    test_codes_x, test_codes_y, test_visit_lens = build_code_xy(test_pids, patient_admission, admission_codes_encoded, max_admission_num, code_num, max_code_num_in_a_visit)


    co_occur_matrix = co_occur(train_pids, patient_admission, admission_codes_encoded, code_num)
    code_code_adj = co_occur_matrix
    train_codes_data = (train_codes_x, train_codes_y, train_visit_lens)
    valid_codes_data = (valid_codes_x, valid_codes_y, valid_visit_lens)
    test_codes_data = (test_codes_x, test_codes_y, test_visit_lens)


    l1 = len(train_pids)
    train_patient_ids = np.arange(0, l1)
    l2 = l1 + len(valid_pids)
    valid_patient_ids = np.arange(l1, l2)
    l3 = l2 + len(test_pids)
    test_patient_ids = np.arange(l2, l3)
    pid_map = dict()
    for i, pid in enumerate(train_pids):
        pid_map[pid] = train_patient_ids[i]
    for i, pid in enumerate(valid_pids):
        pid_map[pid] = valid_patient_ids[i]
    for i, pid in enumerate(test_pids):
        pid_map[pid] = test_patient_ids[i]

    mimic3_path = os.path.join('data', 'mimic3')
    encoded_path = os.path.join(mimic3_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    pickle.dump(pid_map, open(os.path.join(encoded_path, 'pid_map.pkl'), 'wb'))

    print('saving standard data ...')
    standard_path = os.path.join(mimic3_path, 'standard')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    pickle.dump({
        'train_codes_data': train_codes_data,
        'valid_codes_data': valid_codes_data,
        'test_codes_data': test_codes_data
    }, open(os.path.join(standard_path, 'codes_dataset.pkl'), 'wb'))
    pickle.dump({
        'code_code_adj': code_code_adj
    }, open(os.path.join(standard_path, 'auxiliary.pkl'), 'wb'))
