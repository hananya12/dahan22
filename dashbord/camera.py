"""
camera.py
----------
Handles locating and connecting to the Iriun Webcam (or any other camera)
on a Windows machine, and provides a safe way to read frames from it.

Iriun Webcam registers itself as a regular virtual camera device (via
DirectShow on Windows), so OpenCV can access it like any other webcam -
the tricky part is knowing WHICH index it landed on, since that can change
between reboots or when other cameras (laptop webcam, other virtual cams,
etc.) are also connected.

This module tries multiple strategies to find the camera reliably:
  1. If `pygrabber` is available, it enumerates all DirectShow devices by
     NAME and looks for "iriun" in the device name. This is the most
     reliable method because it does not depend on index ordering.
  2. If that fails (pygrabber not installed, or Iriun not found by name),
     it falls back to scanning camera indexes 0..9, opening each one and
     checking whether it actually returns a valid frame.

If no camera can be found at all, `connect()` raises CameraNotFoundError
with a clear, actionable message instead of letting the program crash.
"""

import cv2

# How many device indexes to probe when falling back to index-scanning.
MAX_CAMERA_INDEXES_TO_SCAN = 10


class CameraNotFoundError(Exception):
    """Raised when no working camera (Iriun or otherwise) can be located."""
    pass


class CameraManager:
    """Locates and manages a connection to the Iriun Webcam."""

    def __init__(self, prefer_name_contains="iriun", device_index: int | None = None):
        """
        prefer_name_contains: substring (case-insensitive) to look for in
        the device name when auto-detecting via pygrabber.
        device_index: optional explicit device index to try first.
        """
        self.prefer_name_contains = prefer_name_contains.lower()
        self.capture = None
        self.camera_index = None
        self.device_index = device_index
        # Tracks whether the currently-open camera_index came from an
        # explicit request (constructor/connect() argument) rather than
        # auto-detection, so connect() can decide whether to trust/reuse it.
        self._explicit_index = device_index is not None

    def _find_by_name(self):
        """
        Try to find the Iriun camera by its device name using pygrabber
        (Windows-only, DirectShow device enumeration).
        Returns the device index, or None if not found / unavailable.
        """
        try:
            from pygrabber.dshow_graph import FilterGraph
        except ImportError:
            print("[camera] 'pygrabber' not installed - skipping name-based detection.")
            return None

        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
        except Exception as e:
            print(f"[camera] Could not enumerate DirectShow devices: {e}")
            return None

        if not devices:
            print("[camera] No DirectShow video devices were found on this system.")
            return None

        for index, name in enumerate(devices):
            print(f"[camera] Found device index {index}: {name}")
            if self.prefer_name_contains in name.lower():
                print(f"[camera] Matched Iriun camera at index {index} ('{name}').")
                return index

        print(f"[camera] No device name contained '{self.prefer_name_contains}'.")
        return None

    def _find_by_scanning(self):
        """
        Fallback: try opening camera indexes 0..N and keep the first one
        that actually returns a valid frame. Returns the index, or None.
        """
        print("[camera] Falling back to scanning camera indexes...")
        for index in range(MAX_CAMERA_INDEXES_TO_SCAN):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            try:
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        print(f"[camera] Index {index} works "
                              f"(resolution: {frame.shape[1]}x{frame.shape[0]}).")
                        return index
            finally:
                cap.release()
        return None

    def connect(self, device_index: int | None = None):
        """
        Locate and open the Iriun camera.
        If device_index is given it will attempt to open that specific index
        first. Otherwise it will try name-based detection then scanning.
        Raises CameraNotFoundError if no working camera can be found.
        Returns self so this can be used as a context manager too.
        """
        # Tracks whether `index` (once resolved below) came from an explicit
        # request (constructor/connect() argument) rather than auto-detection.
        explicit = False

        # Explicit device index (parameter) takes highest precedence
        if device_index is not None:
            try_index = device_index
            cap = cv2.VideoCapture(try_index, cv2.CAP_DSHOW)
            try:
                if cap.isOpened():
                    ok, frame = cap.read()
                    cap.release()
                    if ok and frame is not None:
                        index = try_index
                        explicit = True
                    else:
                        index = None
                else:
                    index = None
            except Exception:
                index = None
        else:
            index = None

        # If no explicit index worked, try stored device_index
        if index is None and self.device_index is not None:
            try_index = self.device_index
            cap = cv2.VideoCapture(try_index, cv2.CAP_DSHOW)
            try:
                if cap.isOpened():
                    ok, frame = cap.read()
                    cap.release()
                    if ok and frame is not None:
                        index = try_index
                        explicit = True
                    else:
                        index = None
                else:
                    index = None
            except Exception:
                index = None

        # Try name-based detection
        if index is None:
            index = self._find_by_name()

        # Fall back to index scanning
        if index is None:
            index = self._find_by_scanning()

        if index is None:
            raise CameraNotFoundError(
                "Could not find the Iriun Webcam.\n"
                "Please check the following:\n"
                "  1) The Iriun Webcam app is installed and RUNNING on your phone.\n"
                "  2) The Iriun Webcam driver/desktop client is installed and running on this PC\n"
                "     (download from https://iriun.com).\n"
                "  3) Your phone and PC are on the same Wi-Fi network (or connected via USB\n"
                "     with USB debugging enabled, depending on your Iriun setup).\n"
                "  4) No other application (Zoom, Teams, another Python script, etc.) is\n"
                "     currently holding the camera open."
            )

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise CameraNotFoundError(
                f"Camera index {index} was detected but could not be opened. "
                "Try closing other apps that might be using it and run again."
            )

        self.capture = cap
        self.camera_index = index
        self._explicit_index = explicit
        print(f"[camera] Connected successfully on index {index}.")
        return self

    def read_frame(self):
        """
        Read a single frame from the camera.
        Returns the frame, or None if the read failed (does NOT raise,
        so the main loop can just skip that frame and try again).
        """
        if self.capture is None:
            raise CameraNotFoundError("Camera is not connected - call connect() first.")

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self):
        """Release the camera device cleanly."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            print("[camera] Camera released.")

    # Allow "with CameraManager() as camera:" usage
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()