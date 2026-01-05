import dataclasses
import enum
import logging
import socket
import pickle

import tyro
import websocket_policy_server
import yaml
import torch
import torch.nn as nn
from lerobot_utils import prepare_training_data, rebuild_128dim_action
from egomimic.model_wrap import polciy_wrapper as policy_wrapper

def set_steps_in_modules(root: nn.Module, value=10, key_substr="num_inference_steps", contain_match=True):
    changed = []
    visited = set()

    def match(name: str) -> bool:
        nl, kl = name.lower(), key_substr.lower()
        return (kl in nl) if contain_match else (nl == kl)

    def make_new_val(old, v):
        if isinstance(old, nn.Parameter):
            t = torch.tensor(v, dtype=old.dtype, device=old.device)
            return nn.Parameter(t, requires_grad=old.requires_grad)
        if isinstance(old, torch.Tensor):
            return torch.tensor(v, dtype=old.dtype, device=old.device)
        return v

    def walk(mod: nn.Module, path: str):
        if id(mod) in visited:
            return
        visited.add(id(mod))

        for attr in list(vars(mod).keys()):
            if not match(attr):
                continue
            try:
                old = getattr(mod, attr)
            except Exception:
                continue
            try:
                new = make_new_val(old, value)
                setattr(mod, attr, new)
                changed.append((f"{path}.{attr}", old, new))
            except Exception:
                pass 

        # 递归到子模块
        for name, child in mod.named_children():
            walk(child, f"{path}.{name}")

    walk(root, root.__class__.__name__)
    return changed


### NOTE: to make the policy server work, you need transform the input to numpy array, and the output is also numpy array.
### The visual_preprocessor is combined into the policy wrapper, so you don't need to preprocess the image before sending it to the server.
def load_policy(policy_path, device):
    ckpt = torch.load(policy_path, map_location=device)
    model = ckpt['hyper_parameters']['robomimic_model']
    # policy = model.nets['policy']
    changes = set_steps_in_modules(model.nets['policy'], value=10, key_substr="num_inference_steps", contain_match=True)

    policy = policy_wrapper(model.nets['policy'], model.data_schematic.norm_stats[6])

    class polciy_wrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        @torch.no_grad()
        def forward(self, image, qpos, cond_dict):
            training_data = prepare_training_data(image[0, 0]/255.0, image[0, 1]/255.0, qpos[0])
            for key, value in training_data.items():
                training_data[key] = value.float().to(device)
            actions = self.policy(training_data)
            # for key, value in actions.items():
            #     actions[key] = value.cpu().numpy()
            # actions = rebuild_128dim_action(actions)
            actions = {
                "action": actions['actions_cartesian'].cpu().numpy(),
            }
            return actions
    my_policy_wrapper = polciy_wrapper(policy)
    my_policy_wrapper.eval().to(device)

    return my_policy_wrapper


def Warmup_policy(policy):
    import numpy as np
    from tqdm import tqdm
    obs = {
        "image": np.random.randint(0, 255, (1, 2, 3, 240, 320), dtype=np.uint8), # [0, 255]
        "qpos": np.random.randn(1, 128).astype(np.float32),
        'cond_dict': {
            "plain_text": ["This is a test prompt for the policy server."]
        }
    }
    for i in tqdm(range(10), desc="Warming up policy"):
        _ = policy(obs["image"], obs["qpos"], obs["cond_dict"])


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    policy_path: str | None = None
    norm_stats_path: str | None = None
    device: str = "cuda"

def main(args: Args) -> None:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    with open(args.norm_stats_path, "rb") as f:
        norm_stats = pickle.load(f)

    policy = load_policy(args.policy_path, args.device)
    Warmup_policy(policy)
    logging.info("Policy loaded successfully.")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=8000,
        metadata={},
        norm_stats=norm_stats['h1_inspire'],
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))