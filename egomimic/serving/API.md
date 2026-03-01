# EgoVerse Policy Server API

## Overview

The policy server exposes inference over WebSocket using msgpack serialization. The protocol is compatible with openpi-style clients.

## Endpoints

- **WebSocket** `ws://host:port`: Main inference endpoint
- **HTTP** `GET /healthz`: Health check (returns 200 OK)

## Protocol

1. **Connect**: Client connects via WebSocket.
2. **Metadata**: Server immediately sends a metadata dict (msgpack):

   ```python
   {
       "methods": ["infer", "infer_batch"],
       "embodiment": "rby1",
       "action_horizon": 10,
       "action_dim": 49,
       "camera_keys": ["front_img_1"],
       "proprio_keys": ["robot0_joint_pos"]
   }
   ```

3. **Loop**: Client sends observation(s), server returns action(s).

## Request: Single inference

Send a single observation dict (msgpack):

| Embodiment     | Required keys                         | Notes                                      |
|----------------|---------------------------------------|--------------------------------------------|
| rby1           | `front_img_1`, `robot0_joint_pos`     | Images: (H,W,3) uint8 BGR; joint pos: (26,) float32. Optional: `hand_left_qpos`, `hand_right_qpos` (model may not use) |
| eva_bimanual   | `front_img_1`, `right_wrist_img`, `left_wrist_img`, `joint_positions` | joint_positions: (14,)                     |
| eva right/left | `front_img_1`, `*_wrist_img`, `joint_positions` | joint_positions: (7,)                     |

## Request: Batch inference

Send a list of observation dicts (msgpack). Server returns a list of result dicts.

## Response

```python
{
    "actions": np.ndarray,   # (1, T, D) or (B, T, D) - action chunk
    "embodiment": str,
    "server_timing": {
        "infer_ms": float,
        "prev_total_ms": float | None
    }
}
```

## Usage

```bash
python egomimic/scripts/serve_policy.py \
    --checkpoint logs/RBY_test/.../checkpoints/last.ckpt \
    --port 8000
```
