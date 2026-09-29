import tensorflow as tf
import matplotlib.pyplot as plt
import random

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

random_index = random.randint(0, len(x_train) - 1)

sample_image = x_train[random_index]
sample_label = y_train[random_index][0]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(3, 3))
plt.imshow(sample_image)
plt.title(f"Label: {class_names[sample_label]} (Index: {random_index})")
plt.axis('off')
plt.show()
