from keras.applications.densenet import DenseNet121
from keras.layers import *
from keras.models import Model,load_model

num_classes = 7
img_shape = (224, 224, 3)
base_model = DenseNet121(include_top=False, weights='imagenet',input_shape= img_shape)
dense1 = base_model.output
t1 = GlobalAveragePooling2D()(dense1)
t = Dense(256, activation='relu')(t1)
t = Dropout(0.2)(t)
t = Dense(64, activation='relu')(t)
t = Dense(num_classes, activation='softmax')(t)
model = Model(inputs=base_model.input, outputs=t)

#model = load_model('./multimodel.h5')
model = load_model('./singlemodel.h5')

from keras.preprocessing import image
img_path = 'E:/some data/UHCS/micrographs/class/total/pearlite/micrograph501.tif'
img = image.load_img(img_path, target_size=(224, 224))
x = image.array_to_img(img)
x = np.expand_dims(x, axis=0)
x = x / 255.
print(x.shape)
import matplotlib.pylab as plt
plt.imshow(x[0])
plt.xticks([])
plt.yticks([])
plt.show()

preds = model.predict(x)
#print(preds.shape)
#print(np.argmax(preds[0]))


a_output = model.output[:, np.argmax(preds[0])]

# The is the output feature map of the `conv5_block16_2_conv` layer,
# the last convolutional layer in DenseNet
last_conv_layer = model.get_layer('conv5_block16_2_conv')

# This is the gradient of the "pred" class with regard to
# the output feature map of `conv5_block16_2_conv`
grads = K.gradients(a_output, last_conv_layer.output)[0]
#print(grads.shape)
# This is a vector of shape, where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `conv5_block16_2_conv`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(32):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)


heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

import cv2

# We use cv2 to load the original image
img = cv2.imread(img_path)

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img

# Save the image to disk
cv2.imwrite('./cam.jpg', superimposed_img)
plt.figure()
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure()
plt.imshow(heatmap)
plt.xticks([])
plt.yticks([])
plt.show()

img2 = cv2.imread('./cam.jpg')
plt.figure()
plt.imshow(img2)
plt.xticks([])
plt.yticks([])
plt.show()