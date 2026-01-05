import flax.nnx as nnx
import vla_internal.openpi.models.pi0 as pi0
import vla_internal.openpi.models.model as _model
import vla_internal.openpi.models.tokenizer as _tokenizer
import jax
import jax.numpy as jnp
import numpy as np
import vla_internal.openpi.shared.nnx_utils as nnx_utils
import abc
import os
import torch

class ConvertToJaxData():
    def __init__(self, tokenizer: _tokenizer.PaligemmaTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, image_data, qpos_data, conditioning_dict):
        prompt = conditioning_dict["plain_text"]
        tokenized_prompt = []
        tokenized_prompt_mask = []
        for i in range(len(prompt)):
            tokens, token_masks = self.tokenizer.tokenize(prompt[i].lower())
            tokenized_prompt.append(tokens)
            tokenized_prompt_mask.append(token_masks)
        tokenized_prompt = np.stack(tokenized_prompt, axis=0)
        tokenized_prompt_mask = np.stack(tokenized_prompt_mask, axis=0)
        image_dict = {
            "cam_left": image_data[:, 0].transpose(0, 2, 3, 1),
            "cam_right": image_data[:, 1].transpose(0, 2, 3, 1),
        }
        image_mask_dict = {
            "cam_left": np.ones(image_data.shape[0], dtype=bool),
            "cam_right": np.ones(image_data.shape[0], dtype=bool),
        }
        data_dict = {
            "image": image_dict,
            "image_mask": image_mask_dict,
            "state": qpos_data,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
        }
        return data_dict

class JaxPolicyWrapper(abc.ABC):
    def __init__(self, model_config: pi0.Pi0Config, checkpoint_dir: str):
        super().__init__()
        print(f"Loading model from {checkpoint_dir}")
        model = model_config.load(_model.restore_params(os.path.join(checkpoint_dir, "params"), dtype=jnp.bfloat16))
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = ConvertToJaxData(_tokenizer.PaligemmaTokenizer(max_len=48))
        self._rng = jax.random.key(0)

    def __call__(self, image, qpos, cond_dict):
        # Convert image to torch float and normalize to [-1, 1]
        image = (image.astype(np.float32) / 127.5) - 1.0
        inputs = self._input_transform(image, qpos, cond_dict)
        inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs))
        actions = np.asarray(actions)
        actions = {
            "action": actions
        }
        return actions