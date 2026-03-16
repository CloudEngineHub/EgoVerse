#!/usr/bin/env python3
"""
Live camera viewer with on-the-fly HDF5 capture.

Controls:
  r / SPACE  - start/stop capture
  n          - increment demo id (when idle)
  p          - decrement demo id (when idle)
  q / ESC    - quit

Notes:
- Only D405 cameras are supported.
- Overlay is only for display (not saved to HDF5).
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import yaml

# Add path for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))

from robot_utils import RateLoop
from stream_d405 import RealSenseRecorder


DEFAULT_FRAME_RATE = 30


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    return config


def resolve_save_resolution(
    config: dict,
    save_width: Optional[int],
    save_height: Optional[int],
) -> Optional[Tuple[int, int]]:
    if (save_width is None) != (save_height is None):
        raise ValueError("Both --save-width and --save-height must be specified together")
    if save_width is not None:
        return (int(save_width), int(save_height))
    if "save_resolution" in config:
        save_cfg = config["save_resolution"]
        if "width" not in save_cfg or "height" not in save_cfg:
            raise KeyError("save_resolution must include 'width' and 'height'")
        return (int(save_cfg["width"]), int(save_cfg["height"]))
    return None


def get_camera_config(config: dict, camera_key: str) -> dict:
    if "cameras" not in config:
        raise KeyError(f"No 'cameras' section in config")
    cameras = config["cameras"]
    if camera_key not in cameras:
        available = list(cameras.keys())
        raise KeyError(f"Camera key '{camera_key}' not found. Available: {available}")
    cam_cfg = cameras[camera_key]
    if "enabled" not in cam_cfg:
        raise KeyError(f"Camera config missing 'enabled' for key '{camera_key}'")
    if not cam_cfg["enabled"]:
        raise ValueError(f"Camera '{camera_key}' is disabled in config")
    if "type" not in cam_cfg:
        raise KeyError(f"Camera config missing 'type' for key '{camera_key}'")
    cam_type = str(cam_cfg["type"]).lower()
    if cam_type != "d405":
        raise ValueError(f"Unsupported camera type '{cam_type}'. Only 'd405' is supported.")
    for field in ["serial_number", "width", "height"]:
        if field not in cam_cfg:
            raise KeyError(f"Camera config missing '{field}' for key '{camera_key}'")
    return cam_cfg


def create_d405_recorder(cam_cfg: dict, frame_rate: int) -> RealSenseRecorder:
    serial = str(cam_cfg["serial_number"])
    width = int(cam_cfg["width"])
    height = int(cam_cfg["height"])
    return RealSenseRecorder(serial, width=width, height=height, fps=frame_rate)


def wait_for_camera(recorder: RealSenseRecorder, timeout: float = 15.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        img = recorder.get_image()
        if img is not None:
            return True
        time.sleep(0.05)
    return False


def draw_overlay(
    frame: "cv2.typing.MatLike",
    lines: list[str],
    status: str,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    x = 10
    y = 25
    line_gap = 8

    for line in lines + [status]:
        (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        top_left = (x - 4, y - text_h - 4)
        bottom_right = (x + text_w + 4, y + baseline + 4)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
        cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), thickness)
        y += text_h + baseline + line_gap


class Hdf5Writer:
    def __init__(
        self,
        output_path: Path,
        camera_key: str,
        frame_rate: float,
        output_size: Tuple[int, int],
    ):
        self.output_path = output_path
        self.camera_key = camera_key
        self.frame_rate = frame_rate
        self.height = int(output_size[0])
        self.width = int(output_size[1])
        self.count = 0
        self._file = h5py.File(str(output_path), "w", rdcc_nbytes=1024**2 * 2)
        self._file.attrs["sim"] = False
        self._file.attrs["frame_rate"] = frame_rate
        self._file.attrs["camera_key"] = camera_key
        obs = self._file.create_group("observations")
        images = obs.create_group("images")
        self._dset = images.create_dataset(
            camera_key,
            (0, self.height, self.width, 3),
            maxshape=(None, self.height, self.width, 3),
            dtype="uint8",
            chunks=(1, self.height, self.width, 3),
        )

    def append(self, frame_rgb: "cv2.typing.MatLike") -> None:
        self._dset.resize(self.count + 1, axis=0)
        self._dset[self.count] = frame_rgb
        self.count += 1

    def close(self) -> None:
        self._file.attrs["num_frames"] = self.count
        self._file.close()


def collect_live(
    config_path: str,
    camera_key: str,
    demo_dir: str,
    frame_rate: int,
    start_id: int,
    save_width: Optional[int],
    save_height: Optional[int],
):
    config = load_config(config_path)
    cam_cfg = get_camera_config(config, camera_key)
    save_resolution = resolve_save_resolution(config, save_width, save_height)

    demo_dir = Path(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing D405 camera '{camera_key}'...")
    recorder = create_d405_recorder(cam_cfg, frame_rate=frame_rate)

    print("Waiting for camera to be ready...")
    if not wait_for_camera(recorder):
        raise RuntimeError("Camera not producing frames")

    first_frame = recorder.get_image()
    if first_frame is None:
        raise RuntimeError("Failed to read initial frame")

    if save_resolution is None:
        out_height, out_width = first_frame.shape[:2]
    else:
        out_width, out_height = save_resolution

    window_name = f"{camera_key} - Live View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nControls:")
    print("  r / SPACE  - start/stop capture")
    print("  k          - increment demo id (when idle)")
    print("  j          - decrement demo id (when idle)")
    print("  q / ESC    - quit")
    print()

    recording = False
    current_id = start_id
    writer: Optional[Hdf5Writer] = None
    session_frames = 0

    try:
        with RateLoop(frequency=frame_rate, verbose=False) as loop:
            for _ in loop:
                frame = recorder.get_image()
                if frame is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    continue

                if recording and writer is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if save_resolution is not None:
                        frame_rgb = cv2.resize(
                            frame_rgb,
                            save_resolution,
                            interpolation=cv2.INTER_CUBIC,
                        )
                    writer.append(frame_rgb)
                    session_frames = writer.count

                display_frame = frame.copy()
                status = "REC" if recording else "READY"
                lines = [
                    f"ID: {current_id}",
                    f"Frame: {session_frames}",
                ]
                draw_overlay(display_frame, lines, status)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("r"), ord(" ")):
                    if not recording:
                        output_path = demo_dir / f"demo_{current_id}.hdf5"
                        if output_path.exists():
                            print(f"[WARN] {output_path} exists. Overwriting.")
                        writer = Hdf5Writer(
                            output_path=output_path,
                            camera_key=camera_key,
                            frame_rate=frame_rate,
                            output_size=(out_height, out_width),
                        )
                        recording = True
                        session_frames = 0
                        print(f"[REC] Started demo_{current_id}.hdf5")
                    else:
                        recording = False
                        if writer is not None:
                            writer.close()
                            print(f"[REC] Saved demo_{current_id}.hdf5 ({writer.count} frames)")
                            writer = None
                        current_id += 1
                        session_frames = 0
                elif key == ord("k"):
                    if recording:
                        print("[INFO] Stop recording before changing ID.")
                    else:
                        current_id += 1
                elif key == ord("j"):
                    if recording:
                        print("[INFO] Stop recording before changing ID.")
                    else:
                        current_id = max(0, current_id - 1)

    except KeyboardInterrupt:
        pass
    finally:
        if recording and writer is not None:
            writer.close()
            print(f"[REC] Saved demo_{current_id}.hdf5 ({writer.count} frames)")
        try:
            recorder.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Live view + capture to HDF5 (D405 only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        required=True,
        help="Camera key from config (e.g., 'front_img_1')",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        required=True,
        help="Directory to save demo_*.hdf5 files",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=DEFAULT_FRAME_RATE,
        help="Target frame rate",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Starting demo ID",
    )
    parser.add_argument(
        "--save-width",
        type=int,
        default=None,
        help="Width to resize frames before saving (optional)",
    )
    parser.add_argument(
        "--save-height",
        type=int,
        default=None,
        help="Height to resize frames before saving (optional)",
    )

    args = parser.parse_args()

    collect_live(
        config_path=args.config,
        camera_key=args.camera_key,
        demo_dir=args.demo_dir,
        frame_rate=args.frame_rate,
        start_id=args.start_id,
        save_width=args.save_width,
        save_height=args.save_height,
    )


if __name__ == "__main__":
    main()
