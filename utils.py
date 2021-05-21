from typing import List, Union

import numpy as np
import paddle
import torch

from model import CLIP
from simple_tokenizer import SimpleTokenizer

_tokenizer = SimpleTokenizer()


def convert_pth_to_paddle(state_dict):
    """
    covert torch state dict to paddle
    """

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    #visual.proj" in state_dict

    state_dict_np = {}
    for k in state_dict.keys():

        k0 = k
        value = state_dict[k].type(torch.float32).cpu().numpy()
        if '.running_mean' in k:
            k = k.replace('.running_mean', '._mean')
        if '.running_var' in k:
            k = k.replace('.running_var', '._variance')
        if 'fc' in k and 'weight' in k:
            value = value.transpose()

        if 'c_proj' in k and 'weight' in k:
            value = value.transpose()
        if 'q_proj' in k and 'weight' in k:
            #  print(k)
            value = value.transpose()
        if 'k_proj' in k and 'weight' in k:
            #  print(k)
            value = value.transpose()
        if 'v_proj' in k and 'weight' in k:
            #   print(k)
            value = value.transpose()
        if 'out_proj' in k and 'weight' in k:
            value = value.transpose()
        if 'transformer.resblocks.' in k and 'out_proj' in k:
            #print(k)
            state_dict_np.update({k.replace('out', 'c'): value})

        if 'transformer.resblocks.' in k and 'in_proj_weight' in k:
            weight = value
            dim = weight.shape[0] // 3
            state_dict_np.update({
                k.split('in_proj_weight')[0] + 'q_proj.weight':
                weight[0:dim, :].T
            })
            state_dict_np.update({
                k.split('in_proj_weight')[0] + 'k_proj.weight':
                weight[dim:dim * 2, :].T
            })
            state_dict_np.update({
                k.split('in_proj_weight')[0] + 'v_proj.weight':
                weight[dim * 2:dim * 3, :].T
            })

        if 'transformer.resblocks.' in k and 'in_proj_bias' in k:
            weight = value
            dim = weight.shape[0] // 3
            state_dict_np.update(
                {k.split('in_proj_bias')[0] + 'q_proj.bias': weight[0:dim]})
            state_dict_np.update({
                k.split('in_proj_bias')[0] + 'k_proj.bias':
                weight[dim:dim * 2]
            })
            state_dict_np.update({
                k.split('in_proj_bias')[0] + 'v_proj.bias':
                weight[dim * 2:dim * 3]
            })

        if 'logit_scale' in k:
            value = np.expand_dims(value, 0)

        state_dict_np.update({k: value})
    return state_dict_np


def tokenize(texts: Union[str, List[str]], context_length: int = 77):
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = paddle.zeros((len(all_tokens), context_length), dtype='int64')

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(
                f"Input {texts[i]} is too long for context length {context_length}"
            )
        result[i, :len(tokens)] = paddle.Tensor(np.array(tokens))

    return result


def build_vit_model():

    model = CLIP(embed_dim=512,
                 image_resolution=224,
                 vision_layers=12,
                 vision_width=768,
                 vision_patch_size=32,
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12)
    return model


def build_rn101_model():
    model = CLIP(embed_dim=512,
                 image_resolution=224,
                 vision_layers=(3, 4, 23, 3),
                 vision_width=64,
                 vision_patch_size=None,
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12)
    return model


def build_rn50_model():
    model = CLIP(embed_dim=1024,
                 image_resolution=224,
                 vision_layers=(3, 4, 6, 3),
                 vision_width=64,
                 vision_patch_size=None,
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12)
    return model
