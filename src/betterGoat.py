import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Hold out validation data from the TRAINING set; keep the test data untouched
x_val, y_val = x_train[45000:], y_train[45000:]
x_train, y_train = x_train[:45000], y_train[:45000]

def conv_block(x, filters, drop):
    for _ in range(2):                       # two convs per block
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)             # halve height/width
    return layers.Dropout(drop)(x)

inputs = layers.Input((32, 32, 3))
x = layers.RandomFlip('horizontal')(inputs)  # augmentation (only active in training)
x = layers.RandomTranslation(0.1, 0.1)(x)
x = conv_block(x, 32, 0.2)
x = conv_block(x, 64, 0.3)
x = conv_block(x, 128, 0.4)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10)(x)                # logits
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    x_train, y_train, epochs=60, batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[
        callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
        callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

print(model.evaluate(x_test, y_test, verbose=2))  # use the test set once, at the end

# plot graphs of accuracy and loss
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("goat_accuracy.png", dpi=300, bbox_inches="tight")
plt.savefig("goat_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

plt.show()