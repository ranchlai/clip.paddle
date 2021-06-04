# OpenAI CLIP implemented in [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)

I have ported the CLIP models and inference code to PaddlePaddle and tested for RN101/ViT/RN50 image encoder and the accompanied transformer text encoder, for whomever is interested. (Note: the RN50 text encoder still under testing, and the others are tested to be  nearly the same as pytorch version. )

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. Read the original [README](./README_orig.md) here. Also
[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) are usefully.

## Approach

![CLIP](CLIP.png)

## Quickstart in PaddlePaddle
Install clip:
``` bash
git clone https://github.com/ranchlai/clip.paddle
cd clip.paddle
pip install -e .
```
The following example demonstrates zero-shot classification using CLIP.

``` python

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

    # build model and load the pre-trained weight.
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


```


## Testing against Pytorch version:
Run test in cpu:
``` sh
python test.py -d "cpu"
```
or
run test in Cpu:
``` sh
python test.py -d "gpu:0"
```
Output:

``` Terminal
Start testing to make sure all models are aligned with official implementation
======testing RN101======
Testing passed for text transformer,err: 1.4901161193847656e-08
Testing passed for RN101 image encoder,err:2.7939677238464355e-09
======testing RN50======
Testing passed for text transformer,err: 8.940696716308594e-08
Testing passed for RN50 image encoder,err:5.122274160385132e-09
======testing vit======
Testing passed for text transformer,err: 2.9802322387695312e-08
Testing passed for ViT image encoder,err: 1.341104507446289e-07
```
