import tensorflow as tf




def medical_codes_loss(y_true, y_pred):

    log_eps = 1e-8
    cross_entropy_patient = -(
            y_true * tf.math.log(y_pred + log_eps) +
            (1. - y_true) * tf.math.log(1. - y_pred + log_eps)
    )

    #likelihood_patient
    loss_patient = tf.reduce_mean(tf.reduce_sum(cross_entropy_patient, axis=1), axis=0)

    return loss_patient




