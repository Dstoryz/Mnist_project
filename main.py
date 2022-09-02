import os

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tensorflow as tf
import random
import zipfile
# if 1:
#     !unzip /content/saved_model.zip -d ./

file_zip = zipfile.ZipFile('C:\\Mnist_project\\saved_model.zip')
file_zip.extractall('C:\\Mnist_project')

file_zip.close()


#
#
# filepath = os.getcwd()

# home = Path.home()
path_model = Path("saved_model", "1")
# print(home)
# print(wave_absolute)


model = tf.keras.models.load_model(path_model)

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

test_x = test_x.reshape(-1, 28, 28, 1).astype(np.float32) / 255


def predict_digit(sample):
    prediction = model(sample[None, ...])[0]
    ans = np.argmax(prediction)

    fig = plt.figure(figsize=(12, 4))

    # Визуализация входного изображения
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(sample[:, :, 0], cmap='gray')
    plt.xticks([]), plt.yticks([])

    # Визуализация распределения вероятностей по классам
    ax = fig.add_subplot(1, 2, 2)
    bar_list = ax.bar(np.arange(10), prediction, align='center')
    bar_list[ans].set_color('g')
    ax.set_xticks(np.arange(10))
    ax.set_xlim([-1, 10])
    ax.grid(True)

    plt.show()

    print('Predicted number: {}'.format(ans))



# idx = random.randint(0, test_x.shape[0])
# print("Введите номер картинки из датасета от 0 до {}".format(test_x.shape[0]))
idx = int(input("Введите номер картинки из датасета от 0 до {}".format(test_x.shape[0])))

sample = test_x[idx, ...]
predict_digit(sample)

print('True Answer: {}'.format(test_y[idx]))