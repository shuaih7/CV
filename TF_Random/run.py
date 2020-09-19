import tensorflow as tf
import cv2, os, sys
from matplotlib import pyplot as plt

# Parameters
brightness_max_delta=0.2
contrast_lower=0.9
contrast_upper=1.1


image = cv2.imread("tiger.jpg", -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_bright = tf.image.random_brightness(image, max_delta=brightness_max_delta)
image_contrast = tf.image.random_contrast(image, lower=contrast_lower, upper=contrast_upper)

with tf.Session() as sess:
    img_b, img_c = sess.run([image_bright, image_contrast])
    plt.subplot(1,3,1), plt.imshow(image), plt.title("Original")
    plt.subplot(1,3,2), plt.imshow(img_b), plt.title("Random Brightness")
    plt.subplot(1,3,3), plt.imshow(img_c), plt.title("Random Contrast")
    plt.show()

print("Done")