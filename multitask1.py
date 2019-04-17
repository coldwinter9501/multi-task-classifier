from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model,Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from keras.applications import Xception
from keras.applications import DenseNet121
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.4,
    height_shift_range=0.4,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'E:/some data/UHCS/micrographs/class/train',
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'E:/some data/UHCS/micrographs/class/test',
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical')


def double_generator(cur_generator, train=True):
    cur_cnt = 0
    while True:
        if train and cur_cnt % 4 == 1:
            # provide same image
            x1, y1 = train_generator.next()
            if y1.shape[0] != batch_size:
                x1, y1 = train_generator.next()
            # print(y1)
            # print(np.sort(np.argmax(y1, 1), 0))
            y1_labels = np.argmax(y1, 1)
            has_move = list()
            last_not_move = list()
            idx2 = [-1 for i in range(batch_size)]

            for i, label in enumerate(y1_labels):
                if i in has_move:
                    continue
                for j in range(i+1, batch_size):
                    if y1_labels[i] == y1_labels[j]:
                        idx2[i] = j
                        idx2[j] = i
                        has_move.append(i)
                        has_move.append(j)
                        break
                if idx2[i] == -1:
                    # same element not found and hasn't been moved
                    if len(last_not_move) == 0:
                        last_not_move.append(i)
                        idx2[i] = i
                    else:
                        idx2[i] = last_not_move[-1]
                        idx2[last_not_move[-1]] = i
                        del last_not_move[-1]
            x2 = list()
            y2 = list()
            for i2 in range(batch_size):
                x2.append(x1[idx2[i2]])
                y2.append(y1[idx2[i2]])
            # print(y2)
            x2 = np.asarray(x2)
            y2 = np.asarray(y2)
            # print(x2.shape)
            # print(y2.shape)
        else:
            x1, y1 = cur_generator.next()
            if y1.shape[0] != batch_size:
                x1, y1 = cur_generator.next()
            x2, y2 = cur_generator.next()
            if y2.shape[0] != batch_size:
                x2, y2 = cur_generator.next()
        same = (np.argmax(y1, 1) == np.argmax(y2, 1)).astype(int)
        # print(np.argmax(y1, 1))
        # print(np.argmax(y2, 1))
        # print(same)
        cur_cnt += 1
        yield [x1, x2], [y1, y2, same]

def eucl_dist(inputs):
    x, y = inputs
    return (x - y)**2

input_tensor = Input(shape=(299, 299, 3))
base_model = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
#plot_model(base_model, to_file='xception_model.png')
base_model.layers.pop()
base_model.outputs = [base_model.layers[-1].output]
base_model.layers[-1].outbound_nodes = []
base_model.output_layers = [base_model.layers[-1]]

feature = base_model
img1 = Input(shape=(299, 299, 3), name='img_1')
img2 = Input(shape=(299, 299, 3), name='img_2')

feature1 = feature(img1)
feature2 = feature(img2)
    # let's add a fully-connected layer
category_predict1 = Dense(7, activation='softmax', name='ctg_out_1')(
    Dropout(0.5)(feature1)
)
category_predict2 = Dense(7, activation='softmax', name='ctg_out_2')(
    Dropout(0.5)(feature2)
)

    # concatenated = keras.layers.concatenate([feature1, feature2])
dis = Lambda(eucl_dist, name='square')([feature1, feature2])
judge = Dense(1, activation='softmax', name='bin_out')(dis)

model = Model(inputs=[img1, img2], outputs=[category_predict1, category_predict2, judge])


for layer in base_model.layers:
    layer.trainable = False

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss={'ctg_out_1': 'categorical_crossentropy',
                    'ctg_out_2': 'categorical_crossentropy',
                    'bin_out': 'binary_crossentropy'},
              metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
history = model.fit_generator(double_generator(train_generator),
                    epochs=100,
                    steps_per_epoch=30,
                    validation_data=double_generator(validation_generator),
                    validation_steps = 6,
                    callbacks=[auto_lr]) # otherwise the generator would loop indefinitely

import matplotlib.pylab as plt
fig = plt.figure()
# #plt.plot(history.history["loss"])
plt.plot(history.history["ctg_out_1_acc"])
#plt.plot(history.history["ctg_out_1_f1"])
# #plt.plot(history.history["val_loss"])
plt.plot(history.history["val_ctg_out_1_acc"])
#plt.plot(history.history["val_ctg_out_1_f1"])
plt.title("Model acc")
plt.xlabel("epoch")
#plt.legend(["acc", "F1", 'val_acc', 'val_F1'], loc="upper left")
#plt.savefig('multi_task train.png')
plt.show()


