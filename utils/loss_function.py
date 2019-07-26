from keras.objectives import *
from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf


# Softmax cross-entropy loss function for pascal voc segmentation
# and models which do not perform softmax.
# tensorlow only
def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def my_weighted_loss(onehot_labels, logits):
    """scale loss based on class weights
    """
    # compute weights based on their frequencies
    beta=tf.reduce_sum(onehot_labels,1);
    beta=tf.reduce_sum(beta,1)
    #beta=tf.count_nonzero(onehot_labels,0)
    beta=K.cast(beta, "float32")/320
    class_weights=[[beta[:,1]], [beta[:,0]]];
    class_weights= tf.transpose(class_weights)
    #class_weights =[1,20]  # set your class weights here
    # computer weights based on onehot labels
    class_weights= tf.expand_dims(class_weights, 1);
    #class_weights=tf.expand_dims(class_weights, 1);
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=[onehot_labels], logits=[logits])
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss

# Softmax cross-entropy loss function for coco segmentation
# and models which expect but do not apply sigmoid on each entry
# tensorlow only
def binary_crossentropy_with_logits(ground_truth, predictions):
    return K.mean(K.binary_crossentropy(ground_truth,
                                        predictions,
                                        from_logits=True),
                  axis=-1)
