import tensorflow as tf
from keras.losses import categorical_crossentropy 

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        ce = categorical_crossentropy(y_true=y_true, y_pred=y_pred)
        p_t = tf.exp(-ce)
        loss = alpha * (1-p_t) ** gamma * ce

        return loss
    
    return loss