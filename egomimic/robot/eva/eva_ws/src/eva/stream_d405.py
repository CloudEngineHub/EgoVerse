import sys
import atexit
import signal
import os
from typing import Optional
import threading
import time
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise ImportError(
        "pyrealsense2 is not installed. Install librealsense Python bindings first."
    ) from e


def list_connected_serials() -> list[str]:
    """
    Utility: list serial numbers of connected RealSense devices.
    """
    ctx = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]


class RealSenseRecorder:
    """
    Stream RGB and depth frames from a specific RealSense device.

    Usage:
        serials = list_connected_serials()
        cam = RealSenseRecorder(serials[0])
        img = cam.get_image()               # np.ndarray (480, 640, 3), dtype=uint8 (BGR)
        depth = cam.get_depth()             # np.ndarray (480, 640), dtype=uint16 (mm)
        cam.stop()
        
    Latency monitoring:
        img, latency_info = cam.get_image_with_latency()
        # latency_info = {
        #     'frame_number': int,           # RealSense frame counter
        #     'capture_time': float,         # When frame was captured (time.time())
        #     'frame_age_ms': float,         # How old the frame is when read
        #     'capture_latency_ms': float,   # Estimated sensor-to-SDK latency
        # }
    """

    def __init__(
        self,
        serial_number: str,
        camera_name: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        realsense_json: Optional[str] = None,
        enable_depth: bool = True,
        auto_exposure: bool = True,
        warmup_frames: int = 5,
    ) -> None:
        self._serial = serial_number
        self._camera_name = camera_name
        self._width = width
        self._height = height
        self._fps = fps
        self._realsense_json = realsense_json
        self._enable_depth = enable_depth
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._latest_image: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Latency tracking
        self._frame_number: int = 0
        self._capture_time: float = 0.0  # time.time() when frame was captured
        self._rs_timestamp_ms: float = 0.0  # RealSense hardware timestamp
        self._rs_timestamp_offset: Optional[float] = None  # Offset to convert RS timestamp to wall time
        self._frames_captured: int = 0
        self._frames_read: int = 0
        self._last_read_frame: int = -1  # Track which frame was last read

        self._config.enable_device(self._serial)
        if self._realsense_json:
            self._load_realsense_json(self._realsense_json)

        self._config.enable_stream(
            rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps
        )
        if self._enable_depth:
            self._config.enable_stream(
                rs.stream.depth, self._width, self._height, rs.format.z16, self._fps
            )

        self._profile = self._pipeline.start(self._config)

        if auto_exposure is not None:
            color_sensor = None
            for s in self._profile.get_device().sensors:
                if s.get_info(rs.camera_info.name).lower().startswith("rgb"):
                    color_sensor = s
                    break
            if color_sensor is not None:
                try:
                    color_sensor.set_option(
                        rs.option.enable_auto_exposure, 1 if auto_exposure else 0
                    )
                except Exception:
                    pass

        # Print post-start camera control status (especially useful after JSON load).
        self._print_camera_control_status()

        for _ in range(max(0, warmup_frames)):
            self._wait_for_color_frame(timeout_ms=2000)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        atexit.register(self.stop)

        self._install_signal_handlers()

    def _load_realsense_json(self, json_path: str) -> None:
        """
        Apply a RealSense advanced-mode JSON configuration to this device.
        """
        resolved_path = os.path.abspath(os.path.expanduser(json_path))
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"RealSense JSON config not found: {resolved_path}"
            )

        ctx = rs.context()
        target_device = None
        for device in ctx.query_devices():
            if device.get_info(rs.camera_info.serial_number) == self._serial:
                target_device = device
                break

        if target_device is None:
            raise RuntimeError(
                f"RealSense device with serial {self._serial} not found "
                f"while loading JSON config: {resolved_path}"
            )

        advanced_mode = rs.rs400_advanced_mode(target_device)
        if not advanced_mode.is_enabled():
            raise RuntimeError(
                f"RealSense device {self._serial} is not in advanced mode, "
                f"cannot load JSON config: {resolved_path}"
            )

        with open(resolved_path, "r", encoding="utf-8") as f:
            json_content = f.read()

        advanced_mode.load_json(json_content)

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _wait_for_color_frame(self, timeout_ms: int = 5000) -> Optional[np.ndarray]:
        """
        Internal helper: wait for frameset, extract color np.ndarray (BGR).
        Returns None on timeout.
        """
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
        except Exception:
            return None

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        img = np.asanyarray(color_frame.get_data())
        # Expect shape (480, 640, 3) with the defaults.
        return img

    def _get_sensor_name(self, sensor) -> str:
        try:
            return sensor.get_info(rs.camera_info.name)
        except Exception:
            return "unknown"

    def _supports_any_control_option(self, sensor) -> bool:
        control_options = (
            rs.option.enable_auto_exposure,
            rs.option.exposure,
            rs.option.enable_auto_white_balance,
            rs.option.white_balance,
        )
        for opt in control_options:
            try:
                if sensor.supports(opt):
                    return True
            except Exception:
                continue
        return False

    def _get_color_control_sensor(self):
        sensors = list(self._profile.get_device().sensors)
        if not sensors:
            return None

        # Prefer explicit RGB name when available.
        for sensor in sensors:
            if self._get_sensor_name(sensor).lower().startswith("rgb"):
                return sensor

        # Fallback: choose the sensor that exposes color control options.
        for sensor in sensors:
            if self._supports_any_control_option(sensor):
                return sensor

        return None

    def _read_sensor_option(self, sensor, option):
        try:
            if not sensor.supports(option):
                return None
            return sensor.get_option(option)
        except Exception:
            return None

    def _print_camera_control_status(self) -> None:
        rgb_sensor = self._get_color_control_sensor()
        cam_label = self._camera_name or "unknown_camera"

        if rgb_sensor is None:
            sensor_names = [
                self._get_sensor_name(sensor)
                for sensor in self._profile.get_device().sensors
            ]
            print(
                f"[RealSenseRecorder] {cam_label} ({self._serial}) "
                "No sensor with AE/AWB controls found; "
                f"available sensors={sensor_names}"
            )
            return

        ae_enabled = self._read_sensor_option(rgb_sensor, rs.option.enable_auto_exposure)
        exposure_val = self._read_sensor_option(rgb_sensor, rs.option.exposure)
        awb_enabled = self._read_sensor_option(rgb_sensor, rs.option.enable_auto_white_balance)
        wb_val = self._read_sensor_option(rgb_sensor, rs.option.white_balance)

        ae_str = "n/a" if ae_enabled is None else ("on" if ae_enabled >= 0.5 else "off")
        awb_str = "n/a" if awb_enabled is None else ("on" if awb_enabled >= 0.5 else "off")
        exposure_str = "n/a" if exposure_val is None else f"{exposure_val:.2f}"
        wb_str = "n/a" if wb_val is None else f"{wb_val:.2f}"

        print(
            f"[RealSenseRecorder] {cam_label} ({self._serial}) controls: "
            f"sensor='{self._get_sensor_name(rgb_sensor)}', "
            f"auto_exposure={ae_str}, exposure={exposure_str}, "
            f"auto_white_balance={awb_str}, white_balance={wb_str}"
        )

    def _capture_loop(self) -> None:
        """
        Background loop to poll frames and update latest image/depth buffers.
        """
        while self._running:
            try:
                frames = self._pipeline.poll_for_frames()
                if not frames:
                    time.sleep(0.001)
                    continue
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if self._enable_depth else None
                if not color_frame and not depth_frame:
                    continue

                capture_time = time.time()
                img = np.asanyarray(color_frame.get_data()) if color_frame else None
                depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None

                with self._lock:
                    if img is not None:
                        # Keep latency metadata anchored to color stream (legacy behavior).
                        frame_number = color_frame.get_frame_number()
                        rs_timestamp_ms = color_frame.get_timestamp()
                        if self._rs_timestamp_offset is None:
                            self._rs_timestamp_offset = capture_time - (
                                rs_timestamp_ms / 1000.0
                            )

                        self._latest_image = img
                        self._frame_number = frame_number
                        self._capture_time = capture_time
                        self._rs_timestamp_ms = rs_timestamp_ms
                        self._frames_captured += 1
                    if depth is not None:
                        self._latest_depth = depth
            except Exception:
                time.sleep(0.01)

    def get_image(self) -> Optional[np.ndarray]:
        """
        Return the most recent color frame (BGR, uint8) or None if not available yet.
        Non-blocking.
        """
        with self._lock:
            self._frames_read += 1
            if self._frame_number != self._last_read_frame:
                self._last_read_frame = self._frame_number
            return self._latest_image

    def get_depth(self, convert_to_meters: bool = False) -> Optional[np.ndarray]:
        """
        Return the most recent depth frame or None if unavailable.

        By default returns raw depth as uint16 millimeters (Z16).
        Set convert_to_meters=True to get float32 meters.
        """
        with self._lock:
            depth = self._latest_depth

        if depth is None:
            return None

        if not convert_to_meters:
            return depth

        if depth.dtype == np.uint16:
            return depth.astype(np.float32) / 1000.0
        if depth.dtype != np.float32:
            return depth.astype(np.float32)
        return depth
    
    def get_image_with_latency(self) -> tuple[Optional[np.ndarray], dict]:
        """
        Return the most recent color frame along with latency information.
        
        Returns:
            tuple: (image, latency_info)
                - image: np.ndarray (BGR, uint8) or None if not available
                - latency_info: dict with:
                    - 'frame_number': RealSense frame counter
                    - 'capture_time': When the frame was captured (time.time())
                    - 'frame_age_ms': How old the frame is when read (ms)
                    - 'capture_latency_ms': Estimated sensor-to-SDK latency (ms)
                    - 'frames_captured': Total frames captured since start
                    - 'frames_read': Total read calls since start  
                    - 'is_new_frame': True if this is a different frame than last read
        """
        read_time = time.time()
        with self._lock:
            self._frames_read += 1
            is_new = self._frame_number != self._last_read_frame
            if is_new:
                self._last_read_frame = self._frame_number
            
            if self._latest_image is None:
                return None, {}
            
            # Calculate latencies
            frame_age_ms = (read_time - self._capture_time) * 1000.0
            
            # Estimate capture latency (time from sensor capture to SDK delivery)
            # This uses the RealSense hardware timestamp
            capture_latency_ms = 0.0
            if self._rs_timestamp_offset is not None:
                estimated_capture_wall_time = (self._rs_timestamp_ms / 1000.0) + self._rs_timestamp_offset
                capture_latency_ms = (self._capture_time - estimated_capture_wall_time) * 1000.0
            
            latency_info = {
                'frame_number': self._frame_number,
                'capture_time': self._capture_time,
                'frame_age_ms': frame_age_ms,
                'capture_latency_ms': capture_latency_ms,
                'frames_captured': self._frames_captured,
                'frames_read': self._frames_read,
                'is_new_frame': is_new,
            }
            
            return self._latest_image, latency_info
    
    def get_latency_stats(self) -> dict:
        """
        Get current latency statistics without reading an image.
        
        Returns:
            dict with latency stats and frame counts
        """
        with self._lock:
            return {
                'frame_number': self._frame_number,
                'frames_captured': self._frames_captured,
                'frames_read': self._frames_read,
                'capture_fps': self._fps,
                'serial': self._serial,
            }

    def stop(self) -> None:
        """
        Stop streaming and release the device.
        """
        self._running = False
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
            self._thread = None
        try:
            self._pipeline.stop()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    serials = list_connected_serials()
    if not serials:
        print("No RealSense devices found.")
    else:
        print("Connected devices:", serials)
        # Just stream the first image for testing purpos
        import os
        from egomimic.robot.robot_utils import RateLoop

        out_dir = "./test_wrist_img"
        frame_idx = 0
        os.makedirs(out_dir, exist_ok=True)
        breakpoint()
        test_wrist_cam = RealSenseRecorder(serials[0])
        with RateLoop(frequency=50, max_iterations=500, verbose=True) as loop:
            for i in loop:
                raw_bgr = test_wrist_cam.get_image()
                if raw_bgr is None:
                    continue
                if raw_bgr is not None:
                    cv2.imwrite(
                        os.path.join(out_dir, f"frame_{frame_idx:06d}.png"), raw_bgr
                    )
                    frame_idx += 1
