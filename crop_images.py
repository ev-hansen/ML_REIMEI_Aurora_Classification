#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Crops REIMEI EISA QuickLook plots to just the plot region
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

from os import listdir
from os.path import isfile, join
from PIL import Image
from tqdm import tqdm

# Default coordinates for corners, (0,0) is the top left corner of an image
MIN_X: int = 85
MAX_X: int = 433
MIN_Y: int = 21
MAX_Y: int = 527

INPUT_DIR: str = "./Files/Uncropped"
OUTPUT_DIR: str = "./Files/Cropped"


def crop_directory(input_directory, output_directory,
                   min_x, max_x, min_y, max_y):
    files = [f for f in listdir(input_directory)
             if isfile(join(input_directory, f))]
    files.sort()
    print(files)
    for file in tqdm(files):
        path: str = f"{input_directory}/{file}"
        print(path)
        with Image.open(path) as img:
            img_cropped = img.crop((min_x, min_y, max_x, max_y))
            new_path: str = f"{OUTPUT_DIR}/{file[:-4]}_cropped.png"
            img_cropped.save(fp=new_path)


crop_directory(INPUT_DIR, OUTPUT_DIR, MIN_X, MAX_X, MIN_Y, MAX_Y)
