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
