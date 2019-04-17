import os
import numpy as np
from keras import Input
#from keras.applications import Xception, InceptionV3
from keras.applications.densenet import DenseNet121
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model


img_shape = (224, 224, 3)
epochs = 100
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.4,
    height_shift_range=0.4,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'E:/some data/UHCS/micrographs/class/train',
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'E:/some data/UHCS/micrographs/class/test',
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical')

num_classes = 7
#early_stopping = EarlyStopping(monitor='val_loss', patience=3)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001,
                            cooldown=0, min_lr=0)

base_model = DenseNet121(include_top=False, weights='imagenet',input_shape= img_shape)
dense1 = base_model.output
t1 = GlobalAveragePooling2D()(dense1)
t = Dense(256, activation='relu')(t1)
t = Dropout(0.2)(t)
t = Dense(64, activation='relu')(t)
out = Dense(num_classes, activation='softmax')(t)
model = Model(inputs=base_model.input, outputs=out)

#file = './singlemodel.h5'
#model.load_weights(file)

for layer in base_model.layers[:-1]:
    layer.trainable = False

from keras import backend as K

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
#optimizer=SGD(lr=0.0001,momentum=0.9)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1])
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy',f1])

model.summary()
history = model.fit_generator(train_generator,
                              epochs=epochs,
                              validation_data=validation_generator,
                              callbacks=[auto_lr])
import matplotlib.pylab as plt
fig = plt.figure()
#plt.plot(history.history["loss"])
plt.plot(history.history["acc"])
plt.plot(history.history["f1"])
#plt.plot(history.history["val_loss"])
plt.plot(history.history["val_acc"])
plt.plot(history.history["val_f1"])
plt.title("Model acc/F1")
plt.xlabel("epoch")
plt.legend(["acc", "F1", 'val_acc', 'val_F1'], loc="upper left")
plt.savefig('train.png')
plt.show()

model.save('./singlemodel.h5')
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
data,labels = load_image(path)
print(data.shape)
print(labels.shape)

pred = model.predict(data, batch_size=32)
y_predict = np.argmax(pred, axis =1)
#print(y_predict)
# from sklearn.metrics import confusion_matrix
# confusion_matrix(labels, y_predict)

from sklearn.metrics import classification_report
target_names = ['martensite', 'network', 'pearlite','pearlite+spheroidite', 'pearlite+widmanstatten','spheroidite','spheroidite+widmanstatten']
print(classification_report(labels, y_predict, target_names=target_names))


## feature embedding
model1 = Model(inputs=base_model.input, outputs=t1)
feature = model1.predict(data,batch_size=32)
print(feature.shape)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca', random_state=2)
tsne_feature = tsne.fit_transform(feature)
plt.figure(figsize=(10,10))
color = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    for i in range(X.shape[0]):
        plt.scatter(X[i,0], X[i,1], c=color[labels[i]], alpha=0.5)

    plt.xticks([])
    plt.yticks([])
    plt.savefig('singlemodel embedding.png')
    plt.show()

plot_embedding(tsne_feature)
