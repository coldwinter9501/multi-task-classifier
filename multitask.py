from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model,Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from keras.applications import Xception
from keras.applications import DenseNet121
batch_size = 64
img_shape = (224, 224, 3)
n_class = 7

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
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'E:/some data/UHCS/micrographs/class/test',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

def pair_generator(cur_generator, batch_size, train=True):
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
        one_hot_same = np.zeros([batch_size, 2])
        one_hot_same[np.arange(batch_size), same] = 1
        # print same
        # print one_hot_same
        # print(np.argmax(y1, 1))
        # print(np.argmax(y2, 1))
        # print(same)
        cur_cnt += 1
        yield [x1, x2], [y1, y2, one_hot_same]

def eucl_dist(inputs):
    x, y = inputs
    return (x - y)**2

base_model = DenseNet121(include_top=False, weights='imagenet',input_shape= img_shape)

img1 = Input(shape=(224, 224, 3), name='img_1')
img2 = Input(shape=(224, 224, 3), name='img_2')

densenet1 = base_model(img1)
densenet2 = base_model(img2)

feature1 = GlobalAveragePooling2D()(densenet1)
feature2 = GlobalAveragePooling2D()(densenet2)
# add a fully-connected layer
dense1 = Dense(256,activation='relu')(feature1)
dense1 = Dense(64,activation='relu')(dense1)
category_predict1 = Dense(n_class, activation='softmax', name='ctg_out_1')(dense1)

dense2 = Dense(256,activation='relu')(feature2)
dense2 = Dense(64,activation='relu')(feature2)
category_predict2 = Dense(n_class, activation='softmax', name='ctg_out_2')(dense2)

# concatenated = keras.layers.concatenate([feature1, feature2])
dis = Lambda(eucl_dist, name='square')([feature1, feature2])
dis = Dense(256,activation='relu')(dis)
dis = Dense(64,activation='relu')(dis)
judge = Dense(2, activation='softmax', name='bin_out')(dis)

model = Model(inputs=[img1, img2], outputs=[category_predict1, category_predict2, judge])

for layer in base_model.layers[:-1]:
    layer.trainable = False

from keras.utils import plot_model
plot_model(model, to_file='multitask_model.png')

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

from keras.optimizers import SGD
#optimizer=SGD(lr=0.0001, momentum=0.9)
model.compile(optimizer='adam',
              loss={'ctg_out_1': 'categorical_crossentropy',
                    'ctg_out_2': 'categorical_crossentropy',
                    'bin_out': 'categorical_crossentropy'},
            loss_weights={
                      'ctg_out_1': 1.,
                      'ctg_out_2': 1.,
                      'bin_out': 0.5},
              metrics=['accuracy',f1])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
history = model.fit_generator(pair_generator(train_generator, batch_size=batch_size),
                    epochs=100,
                    steps_per_epoch=30,
                    validation_data=pair_generator(validation_generator, train=False, batch_size=batch_size),
                    validation_steps =20,
                    callbacks=[auto_lr])

model.save('./multitask_model.h5')
#
import matplotlib.pylab as plt
fig = plt.figure()
# #plt.plot(history.history["loss"])
plt.plot(history.history["ctg_out_1_acc"])
plt.plot(history.history["ctg_out_1_f1"])
# #plt.plot(history.history["val_loss"])
plt.plot(history.history["val_ctg_out_1_acc"])
plt.plot(history.history["val_ctg_out_1_f1"])
plt.title("Model acc/F1")
plt.xlabel("epoch")
plt.legend(["acc", "F1", 'val_acc', 'val_F1'], loc="upper left")
plt.savefig('multi_task train.png')
plt.show()


import os
#load data
from keras.preprocessing import image
def load_image(path):
    datalist = []
    labelist = []
    folder = os.listdir(path)
    for i in range(len(folder)):
        img_path = os.listdir(path + '/' + folder[i])
        for j in range(len(img_path)):
            finalpath = path + '/' + folder[i] + '/' + img_path[j]
            img = image.load_img(finalpath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = x / 255.0
            datalist.append(x)
            label = int(i)
            labelist.append(label)
    data = np.array(datalist)
    labels = np.array(labelist)
    return data, labels

path = 'E:/some data/UHCS/micrographs/class/total'
data, labels = load_image(path)
print(data.shape)
print(labels.shape)

model2 = Model(inputs=img1, outputs=category_predict1)
pred = model2.predict(data, batch_size=32)
y_predict = np.argmax(pred, axis=1)

from sklearn.metrics import classification_report
target_names = ['martensite', 'network', 'pearlite','pearlite+spheroidite', 'pearlite+widmanstatten','spheroidite','spheroidite+widmanstatten']
print(classification_report(labels, y_predict,target_names=target_names))

from sklearn.metrics import confusion_matrix
#print(confusion_matrix(labels, y_predict,target_names=target_names))
#
model1 = Model(inputs=img1, outputs=feature1)
featurevec = model1.predict(data, batch_size=32)
print(featurevec.shape)
import matplotlib.pylab as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
tsne = TSNE(n_components=2, init='pca', random_state=2)
tsne_feature = tsne.fit_transform(featurevec)
# pca = PCA(n_components=2)
# pca_feature = pca.fit_transform(featurevec)


def plot_embedding(X, title=None):
    plt.figure(figsize=(10, 10))
    color = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    for i in range(X.shape[0]):
        #plt.text(X[i, 0], X[i, 1],s=str(labels[i]), fontsize=20, color=color[labels[i]])
        plt.scatter(X[i, 0], X[i, 1], c=color[labels[i]], alpha=0.5)
    #plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.savefig('multi_task embedding.png')
    plt.show()

plot_embedding(tsne_feature)
# plot_embedding(pca_feature)
