import websocket_client_policy
import numpy as np

# you should first run "python serve_policy.py --policy_path {} --policy_config_path {} --norm_stats_path {}" to start the server.

# This is the ip address of batiquitos serves, change it to the ip address of your server.
client = websocket_client_policy.WebsocketClientPolicy(host="169.228.49.5", port=8000)

while True:
    ## image normalization is done in the policy wrapper.
    obs = {
        "image": np.random.randint(0, 255, (1, 2, 3, 240, 320), dtype=np.uint8), # [0, 255]
        "qpos": np.random.randn(1, 128).astype(np.float32),
        'cond_dict': {
            "plain_text": ["This is a test prompt for the policy server."]
        }
    }
    action = client.infer(obs)
    print("Action received from server:", action['action'].shape) # B, T, C
    print(f"Received action: {action['server_timing']}")
