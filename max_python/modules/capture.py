import win32gui
import winrt.windows.graphics.capture as graphics_capture
import winrt.windows.graphics.directx as directx
import winrt.windows.graphics.directx.direct3d11 as d3d11
import comtypes
import ctypes
import numpy as np
import cv2

from ctypes import POINTER, cast
from comtypes import GUID
from comtypes.client import CreateObject

# Direct3D11 helpers
_D3D11_DEVICE = None


def _create_d3d_device():
    """Create a D3D11 device for Windows Graphics Capture"""
    global _D3D11_DEVICE
    if _D3D11_DEVICE:
        return _D3D11_DEVICE

    d3d11 = comtypes.client.CreateObject("Direct3D11.Device")
    _D3D11_DEVICE = d3d11
    return d3d11


def _create_capture_item_for_hwnd(hwnd):
    """Create a capture item for a given window handle"""
    factory = graphics_capture.GraphicsCaptureItemInterop()
    interop = comtypes.client.CreateObject("Windows.Graphics.Capture.Interop.GraphicsCaptureItem")
    capture_item = factory.CreateForWindow(hwnd, interop)
    return capture_item


def _hwnd_from_title(title: str):
    """Case-insensitive search for a window title"""
    target = title.lower()
    hwnd_found = None

    def _enum_callback(hwnd, _):
        nonlocal hwnd_found
        if win32gui.IsWindowVisible(hwnd):
            if target in win32gui.GetWindowText(hwnd).lower():
                hwnd_found = hwnd
                return False
        return True

    win32gui.EnumWindows(_enum_callback, None)
    return hwnd_found


def capture_window(title="Graphwar"):
    """Capture a window using Windows.Graphics.Capture API (OBS-style)."""
    hwnd = _hwnd_from_title(title)
    if not hwnd:
        print(f"[capture] Window '{title}' not found.")
        return None

    # Create Direct3D device and capture item
    d3d_device = _create_d3d_device()
    item = _create_capture_item_for_hwnd(hwnd)

    if not item:
        print("[capture] Could not create capture item.")
        return None

    # Create frame pool and capture session
    size = item.Size
    frame_pool = graphics_capture.Direct3D11CaptureFramePool.Create(
        d3d_device, directx.DirectXPixelFormat.B8G8R8A8UIntNormalized, 1, size
    )
    session = frame_pool.CreateCaptureSession(item)
    session.StartCapture()

    # Get a frame
    frame = frame_pool.TryGetNextFrame()
    if not frame:
        print("[capture] No frame captured.")
        return None

    # Convert to numpy array
    surface = frame.Surface
    width, height = size.Width, size.Height
    arr = np.frombuffer(surface, dtype=np.uint8).reshape((height, width, 4))
    arr = arr[:, :, :3]  # drop alpha

    # Clean up
    frame.Close()
    session.Close()
    frame_pool.Close()

    return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
