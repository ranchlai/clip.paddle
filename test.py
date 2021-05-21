import os

import numpy as np
import paddle

from utils import (build_rn50_model, build_rn101_model, build_vit_model,
                   tokenize)

EPS = 1e-5
RN101_OUT_MEAN = 0.01115486491471529
RN50_OUT_MEAN = 0.004151661414653063
VIT_OUT_MEAN = 0.24539007246494293

RN101_TEXT_MEAN = 0.1773824542760849
RN50_TEXT_MEAN = 0.20489279925823212
VIT_TEXT_MEAN = 0.2554764747619629

#ViT-B/32 0.2554764747619629
#RN50 0.20489279925823212
#RN101 0.1773824542760849

print('Start testing to make sure all models are aligned with official impl')

print('======testing RN101======')

model = build_rn101_model()
sd = paddle.load('./assets/RN101.pdparams')
model.load_dict(sd)
image_input = np.load('./assets/image.npy')
image_input = paddle.Tensor(image_input)
model.eval()

text = tokenize(["a diagram", "a dog", "a cat"])
text_feature = model.encode_text(text)
if abs(float(paddle.mean(text_feature**2)) - RN101_TEXT_MEAN) < EPS:
    print('Testing passed for text transformer')
else:
    print('Testing failed for text transformer')

out = model.encode_image(image_input)
if abs(float(paddle.mean(out**2)) - RN101_OUT_MEAN) < EPS:
    print('Testing passed for RN101 image encoder')
else:
    print('Testing failed for RN101 image encoder')

print('======testing RN50======')

model = build_rn50_model()
sd = paddle.load('./assets/RN50.pdparams')
model.load_dict(sd)
image_input = np.load('./assets/image.npy')
image_input = paddle.Tensor(image_input)
model.eval()

text = tokenize(["a diagram", "a dog", "a cat"])
text_feature = model.encode_text(text)
if abs(float(paddle.mean(text_feature**2)) - RN50_TEXT_MEAN) < EPS:
    print('Testing passed for text transformer')
else:
    print('Testing failed for text transformer')

out = model.encode_image(image_input)
if abs(float(paddle.mean(out**2)) - RN50_OUT_MEAN) < EPS:
    print('Testing passed for RN50 image encoder')
else:
    print('Testing failed for RN50 image encoder')

print('======testing vit======')
model = build_vit_model()
sd = paddle.load('./assets/ViT-B-32.pdparams')
model.load_dict(sd)
image_input = np.load('image.npy')
image_input = paddle.Tensor(image_input)
model.eval()

text = tokenize(["a diagram", "a dog", "a cat"])
text_feature = model.encode_text(text)
if abs(float(paddle.mean(text_feature**2)) - VIT_TEXT_MEAN) < EPS:
    print('Testing passed for text transformer')
else:
    print('Testing failed for text transformer')

out = model.encode_image(image_input)
if abs(float(paddle.mean(out**2)) - VIT_OUT_MEAN) < EPS:
    print('Testing passed for ViT image encoder')
else:
    print('Testing failed for ViT image encoder')
