import re


import numpy as np



def encode_code(admission_codes: dict) -> (dict, dict):
    print('encoding code ...')
    code_map = dict()
    for i, (admission_id, codes) in enumerate(admission_codes.items()):
        for code in codes:
            if code not in code_map:
                if len(code_map) + 1 == 33 or len(code_map) + 1 == 78 or len(code_map) + 1 == 111:
                    print(len(code_map) + 1,":",code)
                code_map[code] = len(code_map) + 1


    admission_codes_encoded = {
        admission_id: [code_map[code] for code in codes]
        for admission_id, codes in admission_codes.items()
    }
    return admission_codes_encoded, code_map


def encode_time_duration(patient_admission: dict) -> dict:
    print('encoding time duration ...')
    patient_time_duration_encoded = dict()
    for pid, admissions in patient_admission.items():
        duration = [0]
        for i in range(1, len(admissions)):
            days = (admissions[i]['adm_time'] - admissions[i - 1]['adm_time']).days
            duration.append(days)
        patient_time_duration_encoded[pid] = duration
    return patient_time_duration_encoded

