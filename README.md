# OpenAI CLIP implemented in [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)

I have ported the CLIP models and inference code to PaddlePaddle and tested for RN101/ViT/RN50 image encoder and the accompanied transformer text encoder, for whomever is interested. (Note: the RN50 text encoder still under testing, and the others are tested to be  nearly the same as pytorch version. )

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. Read the original [README](./README.md.orig) here. Also
[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) are usefully.

## Approach

![CLIP](CLIP.png)

## Usage in PaddlePaddle
First install PaddlePaddle and other small dependencies,
``` bash
$ pip install paddlepaddle-gpu
$ pip install ftfy regex tqdm
```

Download pre-trained weights and put them under the folder [./assets](./assets)
- [ViT](https://pan.baidu.com/s/1_RRVAmKg1L9_9SitaA0SAA): v6pp
- [RN101](https://pan.baidu.com/s/1FFzLs_SogVW_OS_LNe_4ZQ): 42mj
- [RN50](https://pan.baidu.com/s/1IKRkYxfpfolklT1_S_u1rQ): 5qrc

The following example demonstrates zero-shot classification using CLIP.

``` python
import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image

from utils import (build_rn50_model, build_rn101_model, build_vit_model,
                   tokenize, transform)

# build model and load the pre-trained weight.
model = build_vit_model()
sd = paddle.load('./assets/ViT-B-32.pdparams')
model.load_dict(sd)
model.eval()

img = Image.open("CLIP.png")
image_input = transform(img)
image_features = model.encode_image(image_input)

text = tokenize(["a diagram", "a dog", "a cat"])
text_features = model.encode_text(text)

# normalized features
image_features = image_features / image_features.norm(axis=-1, keepdim=True)
text_features = text_features / text_features.norm(axis=-1, keepdim=True)

# cosine similarity as logits
logit_scale = model.logit_scale.exp()
logits_per_image = logit_scale * image_features @ text_features.t()
logits_per_text = logit_scale * text_features @ image_features.t()

probs = F.softmax(logits_per_image).numpy()
print(probs)  ## prints: [[0.99299157 0.00484808 0.00216033]]


```


## Testing against Pytorch version: 
``` sh
python test.py
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

