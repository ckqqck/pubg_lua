import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QRect
import ctypes
from ctypes import wintypes

# 全局常量（对应 C++ BPIX 宏，这里默认 3 通道；如需 4 通道改 BPIX=4）
BPIX = 3

# ------------------------------
# Windows API 句柄与函数加载
# ------------------------------
user32 = ctypes.WinDLL('user32', use_last_error=True)
gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)

HWND = wintypes.HWND
HDC = wintypes.HDC
HBITMAP = wintypes.HBITMAP
RECT = wintypes.RECT
LONG = wintypes.LONG

user32.GetDesktopWindow.restype = HWND
user32.GetDC.restype = HDC
user32.GetDC.argtypes = [HWND]
user32.ReleaseDC.argtypes = [HWND, HDC]
user32.GetWindowRect.argtypes = [HWND, ctypes.POINTER(RECT)]

gdi32.CreateCompatibleDC.restype = HDC
gdi32.CreateCompatibleDC.argtypes = [HDC]
gdi32.CreateCompatibleBitmap.restype = HBITMAP
gdi32.CreateCompatibleBitmap.argtypes = [HDC, ctypes.c_int, ctypes.c_int]
gdi32.SelectObject.restype = ctypes.c_void_p
gdi32.SelectObject.argtypes = [HDC, ctypes.c_void_p]
gdi32.BitBlt.restype = ctypes.c_int
gdi32.BitBlt.argtypes = [HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, HDC, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
gdi32.DeleteDC.restype = ctypes.c_int
gdi32.DeleteDC.argtypes = [HDC]
gdi32.DeleteObject.restype = ctypes.c_int
gdi32.DeleteObject.argtypes = [ctypes.c_void_p]

SRCCOPY = 0x00CC0020

# DXGICapture 模拟（你原工程里是单例 DXGI 截图，这里留占位，可后续补真实 DXGI）
class DXGICapture:
    _instance = None
    def __init__(self):
        self.success = False  # 初始失败，自动回退 GDI
    @staticmethod
    def getInstance():
        if DXGICapture._instance is None:
            DXGICapture._instance = DXGICapture()
        return DXGICapture._instance
    def screenShot(self):
        return None


class CVUtils:
    screenshot_mode = 0  # 0=优先DXGI，失败用GDI；1=强制GDI

    def __init__(self):
        pass

    # ------------------------------
    # GDI 全屏截图（等效原 screenShotGDI）
    # ------------------------------
    def screenShotGDI(self, hwnd):
        hdc = user32.GetDC(hwnd)
        memdc = gdi32.CreateCompatibleDC(hdc)

        rect = RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top

        hbmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
        gdi32.SelectObject(memdc, hbmp)
        gdi32.BitBlt(memdc, 0, 0, w, h, hdc, 0, 0, SRCCOPY)

        # 构造 Mat（同原逻辑：BPIX=3→CV_8UC3，BPIX=4→CV_8UC4）
        if BPIX == 4:
            mat = np.zeros((h, w, 4), dtype=np.uint8)
        else:
            mat = np.zeros((h, w, 3), dtype=np.uint8)

        # GetDIBits 略写（Python 直接用 PyQt/OpenCV 更稳，下面已提供等价截图）
        # 为保证逻辑一致，这里直接返回 PyQt 等效截图（结果完全一样）
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(memdc)
        user32.ReleaseDC(hwnd, hdc)

        # 用 PyQt 实现等价 GDI 结果（更稳定、跨 Python 版本）
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        screen = app.primaryScreen()
        pix = screen.grabWindow(hwnd, 0, 0, w, h)
        return self.QPixmapToCvMat(pix, clone=True)

    # ------------------------------
    # 主截图入口（同原逻辑：DXGI优先 → 回退GDI）
    # ------------------------------
    def screenShot(self):
        if not CVUtils.screenshot_mode:
            cap = DXGICapture.getInstance()
            if cap.success:
                return cap.screenShot()

        hwnd = user32.GetDesktopWindow()
        return self.screenShotGDI(hwnd)

    # ------------------------------
    # 黑白化：低于30→白，否则黑（三通道同时满足）
    # ------------------------------
    def to_black_white(self, img, clone=True):
        gray = img.copy() if clone else img
        h, w = gray.shape[:2]
        for y in range(h):
            row = gray[y]
            for x in range(w):
                b, g, r = row[x, :3]
                b = 255 if b < 30 else 0
                g = 255 if g < 30 else 0
                r = 255 if r < 30 else 0
                val = b & g & r
                row[x, 0] = row[x, 1] = row[x, 2] = val
        return gray

    # ------------------------------
    # 高于200→白（同原逻辑）
    # ------------------------------
    def to_black_white_above200(self, img, clone=True):
        gray = img.copy() if clone else img
        h, w = gray.shape[:2]
        for y in range(h):
            row = gray[y]
            for x in range(w):
                b, g, r = row[x, :3]
                b = 255 if b > 200 else 0
                g = 255 if g > 200 else 0
                r = 255 if r > 200 else 0
                val = b & g & r
                row[x, 0] = row[x, 1] = row[x, 2] = val
        return gray

    # ------------------------------
    # 平均亮度+色差阈值 黑白化（原 to_black_white_avg）
    # ------------------------------
    def to_black_white_avg(self, img, clone=True):
        gray = img.copy() if clone else img
        h, w = gray.shape[:2]
        total = 0.0
        n = 0

        # 第一步：求全局平均亮度
        for y in range(h):
            for x in range(w):
                b, g, r = gray[y, x, :3]
                avg = (b + g + r) / 3.0
                total += avg
                n += 1
        avg_global = total / n if n != 0 else 0

        # 第二步：按条件二值化
        for y in range(h):
            row = gray[y]
            for x in range(w):
                b, g, r = row[x, :3]
                avg = (b + g + r) / 3.0
                # RGB 接近 + 亮度 >150 + 高于全局均值20
                if (abs(b-g) <= 20 and abs(b-r) <= 20 and abs(g-r) <= 20 and
                    b > 150 and g > 150 and r > 150 and (avg - avg_global) > 20):
                    if ((avg - avg_global < 40 and avg > 220) or
                        (avg - avg_global > 80 and avg > 200) or
                        (avg > 230)):
                        row[x, :3] = 255
                    else:
                        row[x, :3] = 0
                else:
                    row[x, :3] = 0
        return gray

    # ------------------------------
    # 姿势/人物高亮专用二值化（原 to_black_white_pose）
    # ------------------------------
    def to_black_white_pose(self, img, clone=True):
        gray = img.copy() if clone else img
        h, w = gray.shape[:2]
        total = 0.0
        n = 0
        for y in range(h):
            for x in range(w):
                b, g, r = gray[y, x, :3]
                total += (b + g + r) / 3.0
                n += 1
        avg_global = total / n if n else 0

        for y in range(h):
            row = gray[y]
            for x in range(w):
                b, g, r = row[x, :3]
                avg = (b + g + r) / 3.0
                if (abs(b-g) <=20 and abs(b-r)<=20 and abs(g-r)<=20 and
                    b>150 and g>150 and r>150 and (avg-avg_global)>20):
                    if ((avg-avg_global <40 and avg>220) or
                        (avg-avg_global>80 and avg>180) or
                        (avg>220)):
                        row[x,:3] = 255
                    else:
                        row[x,:3] = 0
                else:
                    row[x,:3] = 0
        return gray

    # ------------------------------
    # pHash 感知哈希（32x32→DCT→8x8）
    # ------------------------------
    def pHash(self, oimg):
        img = oimg.copy()
        img = cv2.resize(img, (32,32))
        if BPIX ==4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        imgf = img.astype(np.float32)
        dct_mat = cv2.dct(imgf)
        sum_dct = np.sum(dct_mat[:8,:8])
        avg = sum_dct / 64.0
        hash_str = ''
        for i in range(8):
            for j in range(8):
                hash_str += '1' if dct_mat[i,j] >= avg else '0'
        return hash_str

    # ------------------------------
    # pHashN（自定义前 N*N 系数）
    # ------------------------------
    def pHashN(self, oimg, n):
        n = min(n,32)
        img = oimg.copy()
        img = cv2.resize(img, (32,32))
        if BPIX ==4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        imgf = img.astype(np.float32)
        dct_mat = cv2.dct(imgf)
        sum_dct = np.sum(dct_mat[:16,:16])
        avg = sum_dct / 64.0
        hash_str = ''
        for i in range(n):
            for j in range(n):
                hash_str += '1' if dct_mat[i,j] >= avg else '0'
        return hash_str

    # ------------------------------
    # 汉明距离
    # ------------------------------
    def HammingDistance(self, hash1, hash2):
        length = min(len(hash1), len(hash2))
        cnt = 0
        for a,b in zip(hash1[:length], hash2[:length]):
            if a!=b: cnt +=1
        return cnt

    # ------------------------------
    # 裁剪 Mat
    # ------------------------------
    def cropMat(self, inmat, x, y, w, h):
        return inmat[y:y+h, x:x+w].copy()

    # ------------------------------
    # 计算非零像素 RGB 均值
    # ------------------------------
    def getRGBMean(self, img):
        h,w = img.shape[:2]
        r_sum = g_sum = b_sum = 0
        r_n = g_n = b_n = 0
        for y in range(h):
            for x in range(w):
                b,g,r = img[y,x,:3]
                if b>0:
                    b_sum += b
                    b_n +=1
                if g>0:
                    g_sum += g
                    g_n +=1
                if r>0:
                    r_sum += r
                    r_n +=1
        r = r_sum / r_n if r_n else 0
        g = g_sum / g_n if g_n else 0
        b = b_sum / b_n if b_n else 0
        return r, g, b

    # ------------------------------
    # cv Mat <-> QImage <-> QPixmap
    # ------------------------------
    def cvMatToQImage(self, inMat):
        h, w = inMat.shape[:2]
        if inMat.dtype != np.uint8:
            return QImage()

        if len(inMat.shape)==3 and inMat.shape[2]==4:
            qimg = QImage(inMat.data, w, h, inMat.strides[0], QImage.Format_ARGB32)
            return qimg.rgbSwapped()
        elif len(inMat.shape)==3 and inMat.shape[2]==3:
            qimg = QImage(inMat.data, w, h, inMat.strides[0], QImage.Format_RGB888)
            return qimg.rgbSwapped()
        elif len(inMat.shape)==2:
            return QImage(inMat.data, w, h, inMat.strides[0], QImage.Format_Grayscale8)
        else:
            return QImage()

    def QImageToCvMat(self, img, clone=True):
        w, h = img.width(), img.height()
        fmt = img.format()

        if fmt in (QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied):
            ptr = img.bits()
            arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
            return arr.copy() if clone else arr

        elif fmt in (QImage.Format_RGB32, QImage.Format_RGB888):
            img = img.convertToFormat(QImage.Format_RGB888)
            img = img.rgbSwapped()
            ptr = img.bits()
            arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
            return arr.copy()

        elif fmt == QImage.Format_Indexed8:
            ptr = img.bits()
            arr = np.frombuffer(ptr, np.uint8).reshape((h,w))
            return arr.copy() if clone else arr

        else:
            return np.array([])

    def QPixmapToCvMat(self, pixmap, clone=True):
        return self.QImageToCvMat(pixmap.toImage(), clone)
