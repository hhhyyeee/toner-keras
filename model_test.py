import numpy as np
import os
import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

model = load_model('model_withAug.h5')
model.summary()

# batch_size = 10
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
#     'data_resized/test',
#     target_size=(300, 300),
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False)

# print('directory: ', test_generator.directory)
# print('classes: ', test_generator.class_indices)
# print('class_mode: ', test_generator.class_mode)
# print('filenames: ', test_generator.filenames)
# print('labels: ', test_generator.labels)

# test_generator.reset() # https://github.com/keras-team/keras/issues/3296#issuecomment-349998110
# output = model.predict(test_generator, steps=4)
# print(test_generator.class_indices)
# for i in range(len(output)):
#     print(test_generator.filenames[i])
#     print(int(output[i][0] * 100))


# 핸드메이드 테스트 셋
aesop_directory = './data_resized/test/aesop/'
kiehls_directory = './data_resized/test/kiehls/'

test_X = []
test_y = []

for filename in os.listdir(aesop_directory):
    img = cv2.imread(aesop_directory + filename, cv2.IMREAD_UNCHANGED)
    if img is not None:
        test_X.append(img)
        test_y.append(0)

for filename in os.listdir(kiehls_directory):
    img = cv2.imread(kiehls_directory + filename, cv2.IMREAD_UNCHANGED)
    if img is not None:
        test_X.append(img)
        test_y.append(1)

# cv2.imshow('img', test_X[25])
# cv2.waitKey(0)
# x = test_X[25].reshape([-1, 300, 300, 3])
# output = model.predict(x)
# print('Output: ', output[0][0], ', Predicted', test_y[25])

for i, x in enumerate(test_X):
    x = x.reshape([-1, 300, 300, 3])
    output = model.predict(x)
    print('Output: ', output[0][0], ', Predicted', test_y[i])