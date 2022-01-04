import cv2
import numpy as np
import os
from glob import glob
import math
from tqdm import tqdm

IMAGE_DIR = 'dataset/label_images_semantic'
OUTPUT_DIR = 'mask'


images = glob(os.path.join(IMAGE_DIR, '*', '*'))
HEIGHT, WIDTH = 2000, 2000

for img_path in tqdm(images):

	op_path = img_path.replace('label_images_semantic', OUTPUT_DIR)
	if not os.path.exists(os.path.split(op_path)[0]):
		os.makedirs(os.path.split(op_path)[0])

	img = cv2.imread(img_path)
	vertical = math.ceil(img.shape[0] / HEIGHT)
	horizontal = math.ceil(img.shape[1] / WIDTH)
	start_v = 0
	count = 1
	for v in range(vertical):
		start_h = 0
		for h in range(horizontal):
			if start_h + WIDTH > img.shape[1]:
				start_h = start_h - ((start_h + WIDTH) - img.shape[1])
			if start_v + HEIGHT > img.shape[0]:
				start_v = start_v - ((start_v + HEIGHT) - img.shape[0])
			crop = img[start_v:start_v+HEIGHT, start_h:start_h+WIDTH]
			cv2.imwrite(op_path.replace('.png', f'_{count}.png'), crop)
			count += 1
			start_h += WIDTH
		start_v += HEIGHT