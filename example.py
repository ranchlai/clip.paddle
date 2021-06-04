import argparse

import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image

from clip import build_model, tokenize, transform

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='gpu:0')
    parser.add_argument('-m', '--model', type=str, default='VIT')
    args = parser.parse_args()

    paddle.set_device(args.device)
    print(f'using device {args.device}')

    # build modeland load the pre-trained weight.
    model = build_model(name=args.model)

    img = Image.open('CLIP.png')
    image_input = transform(img)
    image_features = model.encode_image(image_input)
    prompt = ['a diagram', 'a dog', 'a cat']
    text = tokenize(prompt)
    text_features = model.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(axis=-1,
                                                          keepdim=True)
    text_features = text_features / text_features.norm(axis=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()

    probs = F.softmax(logits_per_image).numpy()[0]
    for i in range(3):
        print(f'{prompt[i]}:\t{probs[i]:.5}')
