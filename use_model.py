#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Script running ML model to classify aurora types from REIMEI data.

This code is adapted from Tensorflow's guide for an image recognition model as
well as Tensorflow's guide to save and load Keras models
See: https://www.tensorflow.org/tutorials/images/classification and
https://www.tensorflow.org/tutorials/keras/save_and_load
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

import tensorflow as tf
from os import listdir
from os.path import isfile, join
import shutil
from tqdm import tqdm

IMAGE_HEIGHT: int = 348
IMAGE_WIDTH: int = 506
IMAGE_SIZE: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
OUTPUT_DIR: str = "./Files/Guessed"
MODEL_DIR: str = "./Models"
VERIFY_DIR: str = "./Files/Verify"
INPUT_DIR: str = "./Files/Unlabeled"
MODEL_DETAILS_PATH = f"{MODEL_DIR}/details.csv"
CATEGORIES: list[str] = [
    "Alfvenic",
    "Diffuse",
    "Inverted_V",
    "Missing",
    "Only_Ions"
    ]
MODELS = [
    tf.keras.models.load_model(f'{MODEL_DIR}/{f}')
    for f in tqdm(listdir(MODEL_DIR)) if (f[-6:] == ".keras")
    ]


def load_as_img_array(file_path: str):
    img = tf.keras.utils.load_img(
        file_path, target_size=IMAGE_SIZE
        )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def predict(models, img_array):
    results: dict[int, int] = {i: 0 for i in range(0, len(CATEGORIES))}
    for model in models:
        category_num = model.predict(img_array).argmax(axis=-1)[0]
        results[category_num] = results[category_num]+1
    max_score = max(results, key=results.get)
    return CATEGORIES[max_score]


def classify_img_array(model, img_array):
    category_num = model.predict(img_array).argmax(axis=-1)[0]
    return CATEGORIES[category_num]


def copy_file(path: str, category: str):
    shutil.copy(src=path, dst=f"{OUTPUT_DIR}/{category}")


def classify_dir(given_dir: str):
    files: list[str] = [f for f in listdir(given_dir) if isfile(join(given_dir, f))]
    files.sort()
    for file in files:
        path: str = f"{given_dir}/{file}"
        img_arr = load_as_img_array(path)
        category = classify_img_array(img_arr)
        print(f"{file}: {category}")
        copy_file(path, category)


def classify_dir_models(models, given_dir: str):
    files: list[str] = [f for f in listdir(given_dir) if isfile(join(given_dir, f))]
    files.sort()
    for file in files:
        path: str = f"{given_dir}/{file}"
        img_arr = load_as_img_array(path)
        category = predict(models, img_arr)
        print(f"{file}: {category}")
        copy_file(path, category)


classify_dir_models(MODELS, VERIFY_DIR)
classify_dir_models(MODELS, INPUT_DIR)
