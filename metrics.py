import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
from sklearn.metrics import f1_score



def evaluate_on_subsets(y, preds, subset_masks, subset_name):
    subset_y = y[subset_masks[subset_name]]
    subset_preds = preds[subset_masks[subset_name]]
    subset_f1 = f1(subset_y, subset_preds)
    subset_prec, subset_recall = top_k_prec_recall(subset_y, subset_preds, ks=[10, 20, 30, 40])
    return subset_f1, subset_prec, subset_recall





def f1(y_true_hot, y_pred, metrics='weighted'):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1
    return f1_score(y_true=y_true_hot, y_pred=result, average=metrics)


def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks), ))
    r = np.zeros((len(ks), ))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / len(t)
    return a / len(y_true_hot), r / len(y_true_hot)


def calculate_occurred(historical, y, preds, ks):
    r1 = np.zeros((len(ks), ))
    r2 = np.zeros((len(ks),))
    n = np.sum(y, axis=-1)
    for i, k in enumerate(ks):
        n_k = n
        pred_k = np.zeros_like(y)
        for T in range(len(pred_k)):
            pred_k[T][preds[T][:k]] = 1
        pred_occurred = np.logical_and(historical, pred_k)
        pred_not_occurred = np.logical_and(np.logical_not(historical), pred_k)
        pred_occurred_true = np.logical_and(pred_occurred, y)
        pred_not_occurred_true = np.logical_and(pred_not_occurred, y)
        r1[i] = np.mean(np.sum(pred_occurred_true, axis=-1) / n_k)
        r2[i] = np.mean(np.sum(pred_not_occurred_true, axis=-1) / n_k)
    return r1, r2


class EvaluateCodesCallBack(Callback):
    def __init__(self, data_gen, y,historical=None):
        super().__init__()
        self.data_gen = data_gen
        self.y = y
        self.historical = historical
        self.f1_score = []
        self.recall10 = []
        self.recall20 = []


    def on_epoch_end(self, epoch, logs=None):
        step_size = len(self.data_gen)
        preds = []
        for i in range(step_size):
            batch_codes_x, batch_visit_lens = self.data_gen[i]
            output = self.model(inputs={
                'visit_codes': batch_codes_x,
                'visit_lens': batch_visit_lens
            }, training=False)
            logits = output#tf.math.sigmoid(output)
            pred = tf.argsort(logits, axis=-1, direction='DESCENDING')
            preds.append(pred.numpy())
        preds = np.vstack(preds)
        f1_score = f1(self.y, preds)
        prec, recall = top_k_prec_recall(self.y, preds, ks=[10, 20, 30, 40])
        if self.historical is not None:
            r1, r2 = calculate_occurred(self.historical, self.y, preds, ks=[10, 20, 30, 40])
            print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall, '\t', 'occurred:', r1, '\t', 'not occurred:', r2)
        else:
            print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall)


        self.f1_score.append(float("%.2f" % (f1_score*100)))
        self.recall10.append(float("%.2f" % (recall[0]*100)))
        self.recall20.append(float("%.2f" % (recall[1]*100)))

        print("f1_score:"+str(self.f1_score))
        print("recall10:" + str(self.recall10))
        print("recall20:" + str(self.recall20))





class EvaluateCodesCallBack_subset(Callback):
    def __init__(self, data_gen, y,subset_mask_test,historical=None):
        super().__init__()
        self.data_gen = data_gen
        self.y = y
        self.historical = historical
        self.f1_score = []
        self.recall10 = []
        self.recall20 = []
        self.subset_mask_test = subset_mask_test


    def on_epoch_end(self, epoch, logs=None):
        step_size = len(self.data_gen)
        preds = []
        outputs = []

        for i in range(step_size):
            batch_codes_x, batch_visit_lens = self.data_gen[i]
            output = self.model(inputs={
                'visit_codes': batch_codes_x,
                'visit_lens': batch_visit_lens
            }, training=False)

            output_temp = output.numpy()
            outputs.append(output_temp)
            logits = output
            pred = tf.argsort(logits, axis=-1, direction='DESCENDING')
            preds.append(pred.numpy())


        preds = np.vstack(preds)
        outputs = np.vstack(outputs)
        f1_score_1 = f1(self.y, preds)
        prec, recall = top_k_prec_recall(self.y, preds, ks=[10, 20, 30, 40])
        if self.historical is not None:
            r1, r2 = calculate_occurred(self.historical, self.y, preds, ks=[10, 20, 30, 40])
            print('\t', 'f1_score:', f1_score_1, '\t', 'top_k_recall:', recall, '\t', 'occurred:', r1, '\t', 'not occurred:', r2)
        else:
            print('\t', 'f1_score:', f1_score_1, '\t', 'top_k_recall:', recall)

        for disease in self.subset_mask_test.keys():
            subset_f1, subset_prec, subset_recall = evaluate_on_subsets(self.y, preds, self.subset_mask_test, disease)

            disease_num = np.sum(self.subset_mask_test[disease])

            print(f'\t {disease}:{disease_num}', f'subset_f1: {subset_f1}',f'\t subset_recall: {subset_recall}')





