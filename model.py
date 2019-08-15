import keras.initializers as KI
import keras.layers as KL
import keras.losses as KLoss
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.utils import conv_utils


class GlobalAttentionPooling2D(Layer):
    def __init__(self, data_format=None, **kwargs):
        super(GlobalAttentionPooling2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[3])
        else:
            return (input_shape[0], input_shape[1])

    def call(self, inputs, **kwargs):
        inputs_s = inputs[0]
        inputs_m = inputs[1]

        shape = tuple(inputs_s.get_shape().as_list())

        outputs_s = tf.multiply(inputs_s, inputs_m)
        outputs_m = K.repeat_elements(inputs_m, shape[-1], axis=-1)

        outputs_s = K.sum(outputs_s, axis=[1, 2])
        outputs_m = K.sum(outputs_m, axis=[1, 2])

        outputs = outputs_s / outputs_m

        return outputs

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(GlobalAttentionPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def spatial_mask_generate(S, y_old, height, width, name='', mask_max=2. / 3, mask_min=1. / 3, init_bias=-0.5,
                          s_mask_max=9. / 10, s_mask_min=7. / 10):
    y_old_w = KL.Lambda(lambda t: K.expand_dims(t, axis=1))(y_old)
    y_old_w = KL.Lambda(lambda t: K.expand_dims(t, axis=1))(y_old_w)
    y_old_w = KL.Lambda(lambda t: K.repeat_elements(t, rep=height, axis=1))(y_old_w)
    y_old_w = KL.Lambda(lambda t: K.repeat_elements(t, rep=width, axis=2))(y_old_w)

    S = KL.Lambda(lambda t: t - K.min(t, axis=-1, keepdims=True))(S)

    M = KL.multiply([S, y_old_w])
    M = KL.Lambda(lambda t: K.sum(t, axis=-1))(M)

    S_and = KL.Lambda(lambda t: K.mean(t, axis=-1, keepdims=True))(S)
    S_and = KL.Conv2D(1, (1, 1), activation='tanh',
                      name='scale_transform_s_and_1' + name, kernel_initializer=KI.Constant(value=1),
                      bias_initializer=KI.Constant(value=init_bias))(S_and)
    S_and = KL.Conv2D(1, (1, 1), activation='sigmoid',
                      name='scale_transform_s_and_2' + name, kernel_initializer=KI.Constant(value=10),
                      bias_initializer=KI.Constant(value=3))(S_and)

    M = KL.Lambda(lambda t: K.expand_dims(t, axis=-1))(M)
    M = KL.multiply([M, S_and])

    M = KL.Conv2D(1, (1, 1), name='scale_transform_1' + name, kernel_initializer=KI.Constant(value=1),
                  bias_initializer=KI.Constant(value=init_bias), activation='tanh')(M)
    M = KL.Conv2D(1, (1, 1), name='scale_transform_2' + name, kernel_initializer=KI.Constant(value=10),
                  bias_initializer=KI.Constant(value=3))(M)  # use median and mean?
    M = KL.Lambda(lambda t: K.sigmoid(t), name='mask' + name)(M)

    M_loss = KL.Lambda(lambda t: spatial_mask_loss(t, max_value=mask_max, min_value=mask_min), name='M_loss' + name)(M)
    S_and_loss = KL.Lambda(lambda t: spatial_mask_loss(t, max_value=s_mask_max, min_value=s_mask_min),
                           name='S_and_loss' + name)(S_and)
    return M, M_loss, S_and_loss


def spatial_mask_loss(mask, max_value=4. / 5, min_value=1. / 3):
    length = K.cast(K.shape(mask)[1] * K.shape(mask)[2], dtype='float32')
    sum_value = K.sum(mask, axis=[1, 2])

    low_loss = K.maximum(min_value * length - sum_value, 0)
    high_loss = K.maximum(sum_value - max_value * length, 0)

    final_loss = high_loss + low_loss
    final_loss = final_loss / length

    return final_loss


def rank_transform(t):
    t = K.stack(t)
    t = tf.transpose(t, [1, 0, 2])
    return t


def rank_loss(y_true, y_pred):
    margin = 0.05
    satisfy = 0.7
    y_pred = tf.transpose(y_pred, [1, 0, 2])
    y_true = tf.squeeze(y_true, axis=[2])
    p1 = y_pred[0]
    p2 = y_pred[1]
    p1 = tf.multiply(p1, y_true)
    p2 = tf.multiply(p2, y_true)
    p1m = K.max(p1, axis=-1)
    p2m = K.max(p2, axis=-1)
    rank1 = K.maximum(0.0, p1m - p2m + margin)
    rank2 = K.maximum(0.0, -(p1m + p2m - 2 * satisfy))
    rank = K.minimum(rank1, rank2)
    return rank


def cross_network_similarity_loss(y_true, y_pred):
    y_pred = tf.transpose(y_pred, [1, 0, 2])
    p1 = y_pred[0]
    p2 = y_pred[1]
    kl = KLoss.kullback_leibler_divergence(p1, p2)
    return tf.maximum(0.0, kl - 0.15)


def entropy(pk):
    pk = pk + 0.00001
    e = -tf.reduce_sum(pk * tf.log(pk), axis=1)
    return e


def entropy_add(t):
    A1 = t[0]
    A2 = t[1]
    pk1 = t[2]
    pk2 = t[3]
    e1 = entropy(pk1)
    e2 = entropy(pk2)
    mp1 = e2 / (e1 + e2)
    mp2 = 1 - mp1
    mp1 = tf.expand_dims(mp1, axis=1)
    mp2 = tf.expand_dims(mp2, axis=1)
    A1 = A1 * mp1
    A2 = A2 * mp2
    A = (A1 + A2) * 2
    return A


def build_global_attention_pooling_model_cascade_attention(base_network, class_num):
    height, width, depth = base_network[0].output_shape[1:]

    feature_map_step_1 = base_network[0].output

    S = Convolution2D(class_num, (1, 1), name='conv_class')(feature_map_step_1)
    A = GlobalAveragePooling2D()(S)

    y_old = KL.Softmax(name='output_1')(A)
    M, M_loss, S_and_loss = spatial_mask_generate(S, y_old, height, width, mask_max=1. / 2, mask_min=1. / 4)

    feature_map_step_2 = base_network[1].output

    S_new = Convolution2D(class_num, (1, 1), name='conv_class_filtered')(feature_map_step_2)
    A_new = GlobalAttentionPooling2D()([S_new, M])

    y_new = KL.Softmax(name='output_2')(A_new)

    r_loss = KL.Lambda(lambda t: rank_transform(t), name='Rank_loss')([y_old, y_new])
    cns_loss = KL.Lambda(lambda t: t, name='Cross_network_similarity_loss')(r_loss)

    x = KL.concatenate([feature_map_step_1, feature_map_step_2])
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(class_num)(x)
    y_all = KL.Softmax(name='output_3')(x)

    A_final = KL.Lambda(lambda t: entropy_add(t))([x, A_new, y_all, y_new])
    # output_5 is the final output
    y = KL.Softmax(name='output_5')(A_final)

    r2_loss = KL.Lambda(lambda t: rank_transform(t), name='Rank_2_loss')([y_old, y])
    r3_loss = KL.Lambda(lambda t: rank_transform(t), name='Rank_3_loss')([y_new, y])

    for layer in base_network[1].layers:
        layer.name += '_2'

    model = Model(inputs=[base_network[0].input, base_network[1].input],
                  outputs=[y_old, y_new, y_all, y, M_loss, S_and_loss,
                           r_loss, cns_loss,
                           r2_loss, r3_loss
                           ])

    model.summary()
    return model
