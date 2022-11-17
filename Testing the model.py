import cv2
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

CATEGORIES = ["Dog", "Cat"]


def prepare(filepath):
    IMG_SIZE = 200
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray')
    plt.show()
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model('dog_cat.model')

prediction = model.predict(
    prepare('')) # Image you want to test
print(CATEGORIES[int(prediction[0][0])])


