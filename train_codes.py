import os
import random
import _pickle as pickle

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import numpy as np

from models.model import ADRL
from loss import medical_codes_loss
from metrics import EvaluateCodesCallBack
from utils import DataGenerator
import warnings

seed = 6669
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


mimic3_path = os.path.join('data', 'mimic3')
encoded_path = os.path.join(mimic3_path, 'encoded')
standard_path = os.path.join(mimic3_path, 'standard')


def load_data() -> (tuple, tuple, dict):
    code_map = pickle.load(open(os.path.join(encoded_path, 'code_map.pkl'), 'rb'))
    codes_dataset = pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))
    auxiliary = pickle.load(open(os.path.join(standard_path, 'auxiliary.pkl'), 'rb'))
    return code_map,codes_dataset, auxiliary


def historical_hot(code_x, code_num):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, x in enumerate(code_x):
        for code in x:
            result[i][code - 1] = 1
    return result


if __name__ == '__main__':

    print(tf.__version__)
    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


    code_map, codes_dataset, auxiliary = load_data()
    train_codes_data, valid_codes_data, test_codes_data = codes_dataset['train_codes_data'], codes_dataset['valid_codes_data'], codes_dataset['test_codes_data']
    (train_codes_x, train_codes_y, train_visit_lens) = train_codes_data
    (valid_codes_x, valid_codes_y, valid_visit_lens) = valid_codes_data
    (test_codes_x, test_codes_y, test_visit_lens) = test_codes_data
    code_code_adj = auxiliary['code_code_adj']

    config = {
        'code_code_adj': tf.constant(code_code_adj, dtype=tf.float32),
        'code_num': len(code_code_adj),
        'patient_num': train_codes_x.shape[0],
        'max_visit_seq_len': train_codes_x.shape[1],
        'output_dim': len(code_map),
        'lambda': 0.3,
        'activation': None
    }

    test_historical = historical_hot(test_codes_x, len(code_map))

    visit_rnn_dims = [200]
    hyper_params = {
        'visit_rnn_dims': visit_rnn_dims,
        'visit_attention_dim': 32,
    }

    test_codes_gen = DataGenerator([test_codes_x, test_visit_lens], shuffle=False)


    def lr_schedule_fn(epoch, lr):
        if epoch < 6:
            lr = 0.003
        elif epoch < 50:
            lr = 0.001
        elif epoch < 200:
            lr = 0.0001
        else:
            lr = 0.00001
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule_fn)
    test_callback = EvaluateCodesCallBack(test_codes_gen, test_codes_y,historical=test_historical)

    checkpoint_path = mimic3_path + "/model_save/model_weight_{epoch:03d}.h5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    adrl_model = ADRL(config, hyper_params)
    adrl_model.compile(optimizer='adam', loss=medical_codes_loss)
    adrl_model.run_eagerly = True

    adrl_model.fit(x={
        'visit_codes': train_codes_x,
        'visit_lens': train_visit_lens
    }, y=train_codes_y.astype(float), validation_data=({
        'visit_codes': valid_codes_x,
        'visit_lens': valid_visit_lens
    }, valid_codes_y.astype(float)), epochs=100, batch_size=32, callbacks=[lr_scheduler, test_callback,cp_callback])
    adrl_model.summary()
