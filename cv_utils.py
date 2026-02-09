import numpy as np
import cv2
import ctypes
from ctypes import wintypes
from recognize_object import RecognizeObject
from config import ANTI_DETECT

import mss
# -------------------------- 修复64位GDI类型定义 --------------------------
# 重新定义64位兼容的GDI类型（核心修复）
user32 = ctypes.WinDLL('user32', use_last_error=True)
gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)

# 修正类型：所有整数参数显式指定为32位（c_int），避免64位溢出
HWND = wintypes.HWND
HDC = wintypes.HDC
HBITMAP = wintypes.HBITMAP
RECT = wintypes.RECT
UINT = wintypes.UINT
INT = ctypes.c_int  # 强制32位int，核心修复点
LONG = wintypes.LONG

# 修正函数原型（所有尺寸参数用INT/c_int，避免64位溢出）
user32.GetDesktopWindow.restype = HWND
user32.GetWindowDC.restype = HDC
user32.GetClientRect.argtypes = [HWND, ctypes.POINTER(RECT)]
user32.GetSystemMetrics.argtypes = [INT]
user32.GetSystemMetrics.restype = INT  # 修复：返回32位int

gdi32.CreateCompatibleDC.restype = HDC
gdi32.CreateCompatibleBitmap.argtypes = [HDC, INT, INT]  # 修复：宽高为32位int
gdi32.SelectObject.argtypes = [HDC, HBITMAP]
gdi32.BitBlt.argtypes = [
    HDC, INT, INT, INT, INT,  # x,y,宽,高（32位int）
    HDC, INT, INT, UINT       # 源x,y,ROP（32位）
]
# 修复GetBitmapBits：第二个参数为ctypes.c_uint（缓冲区长度），避免溢出
gdi32.GetBitmapBits.argtypes = [HBITMAP, ctypes.c_uint, ctypes.c_void_p]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
user32.ReleaseDC.argtypes = [HWND, HDC]

SRCCOPY = 0x00CC0020

class CVUtils:
    screenshot_mode = 2  # 强制GDI截图模式
    BPIX = 3  # 像素通道数

    @staticmethod
    def screen_shot() -> np.ndarray | None:
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screen = sct.grab(monitor)
                # 返回和原代码完全一致的BGR格式numpy数组
                cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)  # 此行仅为格式示意，实际可直接删，np.array后已是BGR
                return np.array(screen, dtype=np.uint8)
        except OverflowError as e:
            print(f"GDI截图溢出错误（64位适配）: {e}")

    @staticmethod
    def screen_shot_gdi(hwnd) -> np.ndarray | None:
        """GDI窗口截图（兼容原有接口）"""
        return CVUtils.screen_shot()

    @staticmethod
    def crop_mat(img: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray | None:
        """裁剪图像 - 带边界检查"""
        if img is None:
            return None
        h, w = img.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        width = max(1, min(width, w - x))
        height = max(1, min(height, h - y))
        return img[y:y + height, x:x + width]

    @staticmethod
    def pHash(img: np.ndarray) -> str:
        """计算图像的pHash值"""
        if img is None:
            return '0' * 64  # 空图像返回全0哈希
        temp_obj = RecognizeObject("", img, 0)
        return temp_obj.phash

    @staticmethod
    def HammingDistance(hash1: str, hash2: str) -> int:
        """计算两个哈希值的汉明距离"""
        if len(hash1) != len(hash2):
            return 64  # 长度不一致返回最大距离
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    @staticmethod
    def to_black_white1(img: np.ndarray, clone: bool = False) -> np.ndarray:
        """转为黑白图（向量化优化）"""
        processed = img.copy() if clone else img
        # 向量化判断，避免逐像素循环
        mask = (processed[:, :, 0] < 30) & (processed[:, :, 1] < 30) & (processed[:, :, 2] < 30)
        processed[mask] = 255
        processed[~mask] = 0
        return processed

    @staticmethod
    def to_black_white_avg(img: np.ndarray, invert: bool = False) -> np.ndarray:
        """基于灰度均值的二值化（武器识别）"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        avg = np.mean(gray)
        _, binary = cv2.threshold(
            gray, avg, 255,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def to_black_white_above200(img: np.ndarray, invert: bool = False) -> np.ndarray:
        """基于200阈值的二值化（武器名称）"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        _, binary = cv2.threshold(
            gray, 200, 255,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)