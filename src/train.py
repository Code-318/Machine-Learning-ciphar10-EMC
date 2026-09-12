import tensorflow as tf
import matplotlib.pyplot as plt

# load the CIFAR-10 db
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Training images:", x_train.shape)
print("Test images:", x_test.shape)

# fix the images
# convert them to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# build CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                           input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# TensorFlow know to train the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train 
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_split=0.1
)

# test the model on test images
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# plot training/validation accuracy
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
# save the figures
plt.savefig("results/figures/baseline_accuracy.png", dpi=300, bbox_inches="tight")
plt.savefig("tests/baseline/baseline_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
