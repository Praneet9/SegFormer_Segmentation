import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch import nn
import streamlit as st


st.title('Semantic Segmentation using SegFormer')
raw_image = st.file_uploader('Raw Input Image')
if raw_image is not None:
	df = pd.read_csv('class_dict_seg.csv')
	classes = df['name']
	palette = df[[' r', ' g', ' b']].values
	id2label = classes.to_dict()
	label2id = {v: k for k, v in id2label.items()}

	image = Image.open(raw_image)
	image = np.asarray(image)
	
	with st.spinner('Loading Model...'):
		feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = SegformerForSemanticSegmentation.from_pretrained("deep-learning-analytics/segformer_semantic_segmentation", 
																 ignore_mismatched_sizes=True,
		                                                         num_labels=len(id2label), id2label=id2label, label2id=label2id,
		                                                         reshape_last_stage=True)
		model = model.to(device)
		model.eval()
	
	with st.spinner('Preparing image...'):
		# prepare the image for the model (aligned resize)
		feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
		pixel_values = feature_extractor_inference(image, return_tensors="pt").pixel_values.to(device)

	with st.spinner('Running inference...'):
		outputs = model(pixel_values=pixel_values)# logits are of shape (batch_size, num_labels, height/4, width/4)

	with st.spinner('Postprocessing...'):
		logits = outputs.logits.cpu()
		# First, rescale logits to original image size
		upsampled_logits = nn.functional.interpolate(logits,
		                size=image.shape[:-1], # (height, width)
		                mode='bilinear',
		                align_corners=False)
		# Second, apply argmax on the class dimension
		seg = upsampled_logits.argmax(dim=1)[0]
		color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3\
		for label, color in enumerate(palette):
		    color_seg[seg == label, :] = color
		# Convert to BGR
		color_seg = color_seg[..., ::-1]
	# Show image + mask
	img = np.array(image) * 0.5 + color_seg * 0.5
	img = img.astype(np.uint8)
	st.image(img, caption="Segmented Image")