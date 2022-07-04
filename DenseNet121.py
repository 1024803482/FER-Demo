import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import cv2
from tensorflow.keras.regularizers import l2
import os
import skimage
import matplotlib.pyplot as plt
import face_recognition

class conv_block(keras.Model):
    def __init__(self, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
        super(conv_block, self).__init__()
        self.nb_filter = nb_filter
        self.bottleneck = bottleneck
        self.dropout_rate = dropout_rate

        self.bn = tf.keras.layers.BatchNormalization()
        # relu
        if self.bottleneck:
            inter_channel = nb_filter * 4
            self.conv1 = keras.layers.Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal',
                                             padding='same', kernel_regularizer=l2(weight_decay))
            self.bn1 = keras.layers.BatchNormalization(epsilon=1.1e-5)
            # relu

        self.conv2 = keras.layers.Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same',
                                         kernel_regularizer=l2(weight_decay))
        if self.dropout_rate:
            self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        out = self.bn(inputs)
        out = tf.nn.relu(out)
        if self.bottleneck:
            out = self.conv1(out)
            out = self.bn1(out)
            out = tf.nn.relu(out)
        out = self.conv2(out)
        if self.dropout_rate:
            out = self.dropout(out)
        return out


class transition_block(keras.Model):
    def __init__(self, nb_filter, compression=1.0, weight_decay=1e-4):
        super(transition_block, self).__init__()
        self.nb_filter = nb_filter
        self.compression = compression
        self.weight_decay = weight_decay
        self.bn1 = tf.keras.layers.BatchNormalization()
        # relu
        self.conv1 = tf.keras.layers.Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal',
                                            padding='same', kernel_regularizer=l2(weight_decay))
        self.avg = tf.keras.layers.AveragePooling2D((2, 2), strides=2, padding='same')

    def call(self, inputs):
        out = self.bn1(inputs)
        out = self.conv1(out)
        out = self.avg(out)
        return out


class dense_block(keras.Model):
    def __init__(self, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                 grow_nb_filters=True):
        super(dense_block, self).__init__()
        self.conv_list = []
        for i in range(nb_layers):
            cb = conv_block(growth_rate, bottleneck, dropout_rate, weight_decay)
            self.conv_list.append(cb)

            if grow_nb_filters:
                nb_filter += growth_rate
        self.nb_filter = nb_filter

    def call(self, inputs):
        x_list = [inputs]
        out = inputs
        for cb in self.conv_list:
            x = cb(out)
            x_list.append(x)
            out = tf.concat([out, x], axis=-1)
        return out

    def get_filter(self):
        return self.nb_filter


def create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                     nb_layers_per_block=[1], bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                     subsample_initial_block=False, activation='softmax'):
    ''' Build the DenseNet model
        Args:
            nb_classes: number of classes
            img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
            include_top: flag to include the final Dense layer
            depth: number or layers
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
            nb_layers_per_block: list, number of layers in each dense block
            bottleneck: add bottleneck blocks
            reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
            dropout_rate: dropout rate
            weight_decay: weight decay rate
            subsample_initial_block: Set to True to subsample the initial convolution and
                    add a MaxPool2D before the dense blocks are added.
            subsample_initial:
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                    Note that if sigmoid is used, classes must be 1.
        Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = -1

    if type(nb_layers_per_block) is not list:
        print('nb_layers_per_block should be a list!!!')
        return 0

    final_nb_layer = nb_layers_per_block[-1]
    nb_layers = nb_layers_per_block[:-1]

    if nb_filter <= 0:
        nb_filter = 2 * growth_rate
    compression = 1.0 - reduction
    if subsample_initial_block:
        initial_kernel = (7, 7)
        initial_strides = (2, 2)
    else:
        initial_kernel = (3, 3)
        initial_strides = (1, 1)

    x = tf.keras.layers.Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
    if subsample_initial_block:
        x = tf.keras.layers.BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    for block_index in range(nb_dense_block - 1):
        model = dense_block(nb_layers[block_index], nb_filter, growth_rate, bottleneck=bottleneck,
                                   dropout_rate=dropout_rate, weight_decay=weight_decay)
        x = model(x)
        nb_filter = model.get_filter()
        x = transition_block(nb_filter, compression=compression, weight_decay=weight_decay)(x)
        nb_filter = int(nb_filter * compression)

    # 最后一个block没有transition_block
    model = dense_block(nb_layers[block_index], nb_filter, growth_rate, bottleneck=bottleneck,
                        dropout_rate=dropout_rate, weight_decay=weight_decay)
    x = model(x)
    nb_filter = model.get_filter()

    x = tf.keras.layers.BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    if include_top:
        x = tf.keras.layers.Dense(nb_classes, activation=activation)(x)

    return x

def face_detect(image, module=1):
    face_locations = face_recognition.face_locations(image)
    num = len(face_locations)
    face = []
    if num:
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face.append(image[top:bottom, left:right])
            if module == 1:
                image = cv2.rectangle(image,(left, top), (right, bottom), (50, 50, 250))
            else :
                image = cv2.rectangle(image,(left, top), (right, bottom), (255))
        return image, face, face_locations
    else :
        return image, None, None

def face_replace(image, location, label, module=1):
    top, right, bottom, left = location
    face = cv2.imread(label+'.png', module)
    face = cv2.resize(face, (right-left,bottom-top))
    if module == 0 :
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    for y in range(bottom-top):
        for x in range(right-left):
            if np.average(face[y, x]) < 210:
                image[top+y, left+x] = face[y, x]
    return image

if __name__ == '__main__':
    inputs = keras.Input(shape=(48,48,1), batch_size=64)
    x = create_dense_net(7, inputs, include_top=True, depth=121, nb_dense_block=4 ,growth_rate=16, nb_filter=-1, nb_layers_per_block=[6, 12, 32, 32], bottleneck=True, reduction=0.5,dropout_rate=0.2, activation='softmax')
    model = tf.keras.Model(inputs, x, name='densenet121')
    model.summary()