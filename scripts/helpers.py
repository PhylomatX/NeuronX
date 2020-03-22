import os
import numpy as np
from morphx.data import basics


def calculate_error(set_path: str, out_path: str, mode: str = 'mv'):
    set_path = os.path.expanduser(set_path)
    out_path = os.path.expanduser(out_path)
    dirs = os.listdir(set_path)
    at80_t = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
              'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
    at100_t = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
               'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
    at32_t = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
              'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
    for di in dirs:
        at80 = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
                'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
        at100 = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
                 'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
        at32 = {'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
                'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
        data = basics.load_pkl(set_path + di + f'/{di[:-21]}_mv.pkl')
        pool = None
        for key in data:
            if 'at80' in key:
                pool = at80
            elif 'at100' in key:
                pool = at100
            elif 'at32' in key:
                pool = at32
            total = data[key]['total'][mode]
            pool['accuracy'].append(total['accuracy'])
            for score in pool['macro avg']:
                pool['macro avg'][score].append(total['macro avg'][score])
            for score in pool['weighted avg']:
                pool['weighted avg'][score].append(total['weighted avg'][score])
        import ipdb
        ipdb.set_trace()
        student_t = 0.6  # 4 measurements
        at80_t['accuracy'].append(student_t * np.std(at80['accuracy']))
        at100_t['accuracy'].append(student_t * np.std(at80['accuracy']))
        at32_t['accuracy'].append(student_t * np.std(at80['accuracy']))
        for key in at80:
            if key == 'accuracy':
                continue
            else:
                for score in at80[key]:
                    at80_t[key][score].append(student_t * np.std(at80[key][score]))
        for key in at100:
            if key == 'accuracy':
                continue
            else:
                for score in at100[key]:
                    at100_t[key][score].append(student_t * np.std(at100[key][score]))
        for key in at32:
            if key == 'accuracy':
                continue
            else:
                for score in at32[key]:
                    at32_t[key][score].append(student_t * np.std(at32[key][score]))

    # average over all 3 validations
    for key in at80_t:
        if key == 'accuracy':
            at80_t[key] = np.mean(at80_t[key])
        else:
            for score in at80_t[key]:
                at80_t[key][score] = np.mean(at80_t[key][score])
    for key in at100_t:
        if key == 'accuracy':
            at100_t[key] = np.mean(at100_t[key])
        else:
            for score in at100_t[key]:
                at100_t[key][score] = np.mean(at100_t[key][score])
    for key in at32_t:
        if key == 'accuracy':
            at32_t[key] = np.mean(at32_t[key])
        else:
            for score in at32_t[key]:
                at32_t[key][score] = np.mean(at32_t[key][score])

    with open(out_path + 'error.txt', 'w') as f:
        f.write(f"Accuracy error: {(at80_t['accuracy'] + at100_t['accuracy'] + at32_t['accuracy']) / 3}\n")
        f.write("\nmacro avg:\n")
        f.write(f"Precision error: {(at80_t['macro avg']['precision'] + at100_t['macro avg']['precision'] + at32_t['macro avg']['precision']) / 3}\n")
        f.write(f"Recall error: {(at80_t['macro avg']['recall'] + at100_t['macro avg']['recall'] + at32_t['macro avg']['recall']) / 3}\n")
        f.write(f"f1 error: {(at80_t['macro avg']['f1-score'] + at100_t['macro avg']['f1-score'] + at32_t['macro avg']['f1-score']) / 3}\n")
        f.write("\nweighted avg:\n")
        f.write(f"Precision error: {(at80_t['weighted avg']['precision'] + at100_t['weighted avg']['precision'] + at32_t['weighted avg']['precision']) / 3}\n")
        f.write(f"Recall error: {(at80_t['weighted avg']['recall'] +at100_t['weighted avg']['recall'] + at32_t['weighted avg']['recall']) / 3}\n")
        f.write(f"f1 error: {(at80_t['weighted avg']['f1-score'] + at100_t['weighted avg']['f1-score'] + at32_t['weighted avg']['f1-score']) / 3}\n")
    f.close()
