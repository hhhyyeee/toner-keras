import os
import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import History
from keras import backend as K
import matplotlib.pyplot as plt

# 모델 구성
batch_size = 10
train_size = 127
validation_size = 38
if K.image_data_format() == 'channels_first':
    input_shape = (3, 300, 300)
else:
    input_shape = (300, 300, 3)

# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

model = Sequential()
model.add(Conv2D(64, 3, activation='relu', input_shape=input_shape))
model.add(Conv2D(64, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, activation='relu', input_shape=input_shape))
model.add(Conv2D(128, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, 3, activation='relu', input_shape=input_shape))
model.add(Conv2D(256, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
# train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data_resized/train',
    target_size=(300, 300),
    batch_size=batch_size,
    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
    'data_resized/validation',
    target_size=(300, 300),
    batch_size=batch_size,
    class_mode='binary')
test_generator = test_datagen.flow_from_directory(
    'data_resized/test',
    target_size=(300, 300),
    batch_size=batch_size,
    class_mode='binary')

print('directory: ', train_generator.directory)
print('classes: ', train_generator.class_indices)
print('class_mode: ', train_generator.class_mode)
print('labels: ', train_generator.labels)

# 학습 시작
history = History()
model.fit(
    train_generator,
    steps_per_epoch=train_size // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_size//batch_size,
    epochs=20,
    callbacks=[history])

model.save('model_vgg_0630.h5')
model.save_weights('weight_vgg_0630.h5')

# 그래프 그리기
print("history.history: ", history.history)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 모델 평가
print(">>> Evaluate <<<")
scores = model.evaluate_generator(test_generator, steps=4)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 테스트 데이터 예측
print(">>> output <<<")
output = model.predict_generator(test_generator, steps=4)
print(test_generator.class_indices)
print(output)

# # 핸드메이드 테스트 셋
# aesop_directory = './data_resized/test/aesop/'
# kiehls_directory = './data_resized/test/kiehls/'

# test_X = []
# test_y = []

# for filename in os.listdir(aesop_directory):
#     img = cv2.imread(aesop_directory + filename, cv2.IMREAD_UNCHANGED)
#     if img is not None:
#         test_X.append(img)
#         test_y.append(0)

# for filename in os.listdir(kiehls_directory):
#     img = cv2.imread(kiehls_directory + filename, cv2.IMREAD_UNCHANGED)
#     if img is not None:
#         test_X.append(img)
#         test_y.append(1)

# for i, x in enumerate(test_X):
#     x = x.reshape([-1, 300, 300, 3])
#     output = model.predict(x)
#     print('Output: ', output[0][0], ', Predicted', test_y[i])