#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Script training ML model to classify aurora types from REIMEI data.

This code is adapted from Tensorflow's guide for an image recognition model.
See: https://www.tensorflow.org/tutorials/images/classification.

The images are quicklook plots of REIMEI ion and electron data, cropped so
that only the contents of the plots are visible and not the axes or any other
information that is in the file.

Currently using a CNN, want to use a ResNet50 model in the future after more
data for training can be validated.
"""

__authors__: list[str] = ["Ev Hansen"]
__contact__: str = "ephansen+gh@terpmail.umd.edu"

__credits__: list[list[str]] = [
    ["Ev Hansen", "Python code"],
    ["Emma Mirizio", "Co-Mentor"],
    ["Marilia Samara", "Co-Mentor"],
    ["Ellen", "Previous work sorting REIMEI plots"]
    ]

__date__: str = "2024/10/26"
__status__: str = "Development"
__version__: str = "0.0.2"
__license__: str = "MIT"

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
from datetime import datetime

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Constants
# cropped REIMEI quicklook plot images are 348 by 506
IMAGE_HEIGHT: int = 348
IMAGE_WIDTH: int = 506
IMAGE_SIZE: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
BATCH_SIZE: int = 1  # Lower batch size is slower but sometimes more accurate
NUM_EPOCHS: int = 12  # I worry model will overfit if I increase this
TRAIN_DIR: str = "./Files/Labeled/Train"
TEST_DIR: str = "./Files/Labeled/Test"
VERIFY_DIR: str = "./Files/Verify"
MODEL_DIR: str = "./Models"
MODEL_DETAILS_PATH = f"{MODEL_DIR}/details.csv"
CATEGORIES: list[str] = [
    "Alfvenic",
    "Diffuse",
    "Inverted_V",
    "Missing",
    "Only_Ions"
    ]
AUTOTUNE = tf.data.AUTOTUNE
RANDOM_SEED = random.randint(0, 999999999)

# Initilaize training data set
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)


# Initilaize validation data set
val_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

# Initialize CNN Model
class_names = train_ds.class_names
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes = len(class_names)
model = Sequential([
    layers.Rescaling(1./255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # layers.Conv2D(32, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(64, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
model.summary()

# Fit the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS
)

# Acquire data from training to plot
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, NUM_EPOCHS+1)

# Plot the model
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xticks(range(0, NUM_EPOCHS+1))

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xticks(range(0, NUM_EPOCHS+1))


# Test the model with handpicked samples from each category
for category in CATEGORIES:
    file_path = f"{VERIFY_DIR}/{category}.png"
    img = tf.keras.utils.load_img(
        file_path, target_size=IMAGE_SIZE
        )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted = class_names[np.argmax(score)]

    correct_or_wrong = "correct" if (predicted == category) else "wrong"

    text_details = "\n".join((
        f"Guess was {correct_or_wrong}",
        f"Predicted category: {predicted}",
        f"Actual category: {category}",
        "{:.2f} percent confidence. \n".format(100 * np.max(score))
    ))
    print(text_details)

train_stats = zip(
    epochs_range,
    acc,
    val_acc,
    loss,
    val_loss
    )
print(list(train_stats))

# Save plot as a PNG file and model with tf format
now = datetime.now()
model_id: str = "{}-{}-{}_{}-{}-{}".format(
    now.year, now.month, now.day, now.hour, now.minute, now.second
)
plot_filename: str = f"{model_id}_training_performance.png"
model.save(f"./{MODEL_DIR}/{model_id}.keras", save_format='keras')
plt.savefig(f"./{MODEL_DIR}/{plot_filename}")

# Add model details to csv file
with open(MODEL_DETAILS_PATH, 'a') as file:
    output_details: str = "{}, {}, {}, {}, {}, {}, {}, {}\n".format(
        model_id, BATCH_SIZE, NUM_EPOCHS, acc[-1], val_acc[-1],
        loss[-1], val_loss[-1], plot_filename)
    print(output_details)
    file.write(output_details)
