import sys
import atexit
import signal
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
    Stream RGB frames (BGR8) at 640x480@30 from a specific RealSense device.

    Usage:
        serials = list_connected_serials()
        cam = RealSenseRecorder(serials[0])
        img = cam.get_image()               # np.ndarray (480, 640, 3), dtype=uint8 (BGR)
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
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        auto_exposure: bool = True,
        warmup_frames: int = 5,
    ) -> None:
        self._serial = serial_number
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._latest_image: Optional[np.ndarray] = None
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

        self._config.enable_stream(
            rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps
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

        for _ in range(max(0, warmup_frames)):
            self._wait_for_color_frame(timeout_ms=2000)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        atexit.register(self.stop)

        self._install_signal_handlers()

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

    def _capture_loop(self) -> None:
        """
        Background loop to poll frames and update the latest image buffer.
        """
        while self._running:
            try:
                frames = self._pipeline.poll_for_frames()
                if not frames:
                    time.sleep(0.001)
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # Capture timestamp immediately
                capture_time = time.time()
                img = np.asanyarray(color_frame.get_data())
                
                # Get RealSense frame metadata
                frame_number = color_frame.get_frame_number()
                rs_timestamp_ms = color_frame.get_timestamp()  # Hardware timestamp in ms
                
                # Compute offset between RS timestamp and wall clock (once)
                if self._rs_timestamp_offset is None:
                    self._rs_timestamp_offset = capture_time - (rs_timestamp_ms / 1000.0)
                
                with self._lock:
                    self._latest_image = img
                    self._frame_number = frame_number
                    self._capture_time = capture_time
                    self._rs_timestamp_ms = rs_timestamp_ms
                    self._frames_captured += 1
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
