# OpenAI CLIP ported to [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md)

See  [original README](./README.md.orig) here


CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.



## Approach

![CLIP](CLIP.png)


## Usage
Install PaddlePaddle and other small dependencies, 
```bash
$ pip install paddlepaddle-gpu
$ pip install ftfy regex tqdm
```

```python
import paddle
from utils import tokenize,build_rn101_model,build_rn50_model,build_vit_model

# build and load model
model = build_rn101_model()
sd = paddle.load('./assets/RN101.pdparams')
model.load_dict(sd)
model.eval()

# load image and load model
image_input = np.load('./assets/image.npy')
image_input = paddle.Tensor(image_input)


image_feature = model.encode_image(image_input)
text = tokenize(["a diagram", "a dog", "a cat"])
text_feature = model.encode_text(text)

```
