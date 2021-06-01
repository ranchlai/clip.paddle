import os

import numpy as np
import paddle

from utils import (build_model,tokenize)

paddle.set_device('gpu:0')
EPS = 1e-5
RN101_OUT_MEAN = 0.01115486491471529
RN50_OUT_MEAN = 0.004151661414653063
VIT_OUT_MEAN = 0.24539007246494293

RN101_TEXT_MEAN = 0.1773824542760849
RN50_TEXT_MEAN = 0.20474494993686676
VIT_TEXT_MEAN = 0.2554764747619629

#ViT-B/32 0.2554764747619629
#RN50 0.20489279925823212
#RN101 0.1773824542760849

print(
    'Start testing to make sure all models are aligned with official implementation'
)

print('======testing RN101======')

model = build_model('RN101')
image_input = np.load('./assets/image.npy')
image_input = paddle.Tensor(image_input)
model.eval()

text = tokenize(["a diagram", "a dog", "a cat"])
text_feature = model.encode_text(text)
err = abs(float(paddle.mean(text_feature**2)) - RN101_TEXT_MEAN)
if err < EPS:
    print(f'Testing passed for text transformer,err: {err}')
else:
    print(f'Testing failed for text transformer, err: {err}')

out = model.encode_image(image_input)
err = abs(float(paddle.mean(out**2)) - RN101_OUT_MEAN)
if err < EPS:
    print(f'Testing passed for RN101 image encoder,err: {err}')
else:
    print(f'Testing failed for RN101 image encoder,err: {err}')

print('======testing RN50======')

model = build_model('RN50')
image_input = np.load('./assets/image.npy')
image_input = paddle.Tensor(image_input)
model.eval()

text_feature = model.encode_text(text)
err = abs(float(paddle.mean(text_feature**2)) - RN50_TEXT_MEAN)
if err < EPS:
    print(f'Testing passed for text transformer,err: {err}')
else:
    print(f'Testing failed for text transformer, err: {err}')

out = model.encode_image(image_input)
err = abs(float(paddle.mean(out**2)) - RN50_OUT_MEAN)
if err < EPS:
    print(f'Testing passed for RN50 image encoder,err: {err}')
else:
    print(f'Testing failed for RN50 image encoder,err: {err}')

print('======testing vit======')
model = build_model(name='ViT-B-32')
image_input = np.load('./assets/image.npy')
image_input = paddle.Tensor(image_input)
model.eval()

text_feature = model.encode_text(text)
err = abs(float(paddle.mean(text_feature**2)) - VIT_TEXT_MEAN)
if err < EPS:
    print(f'Testing passed for text transformer,err: {err}')
else:
    print(f'Testing failed for text transformer, err: {err}')

out = model.encode_image(image_input)
err = abs(float(paddle.mean(out**2)) - VIT_OUT_MEAN)
if err < EPS:
    print(f'Testing passed for ViT image encoder,err: {err}')
else:
    print(f'Testing failed for ViT image encoder,err: {err}')
