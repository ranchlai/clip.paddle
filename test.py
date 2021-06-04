import argparse
import os

import numpy as np
import paddle
from PIL import Image

from clip import build_model, tokenize, transform

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='gpu:0')
    args = parser.parse_args()

    print(f'using device {args.device}')
    paddle.set_device(args.device)
    EPS = 1e-4
    RN101_OUT_MEAN = 0.01163482666015625
    RN50_OUT_MEAN = 0.003353118896484375
    VIT_OUT_MEAN = 0.2117919921875

    RN101_TEXT_MEAN = 0.1773824542760849
    RN50_TEXT_MEAN = 0.20474494993686676
    VIT_TEXT_MEAN = 0.2554764747619629

    print(
        'Start testing to make sure all models are aligned with official implementation'
    )
    img = Image.open('CLIP.png')
    image_input = transform(img)

    print('======testing RN101======')
    model = build_model('RN101')
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
    model = build_model(name='VIT')
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
