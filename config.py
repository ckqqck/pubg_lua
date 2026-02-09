import os
import sys
import random
import time

# 按键映射表（保留）
KEY_CODE_MAP = {9: "tab", 120: "f9", 121: "f10", 49: "1", 50: "2", 52: "4", 53: "5"}

# 【新增】抗检测核心配置
ANTI_DETECT = {
    "RECOG_FPS": random.randint(5, 8),  # 随机识别帧率5~8FPS
    "SCREEN_DELAY": random.uniform(0.1, 0.2),  # 截图延迟
    "KEY_DELAY": (0.05, 0.15),  # 按键防抖50~150ms
    "THREAD_SLEEP": (0.008, 0.015),  # 线程休眠8~15ms
    "HASH_THRESHOLD": 10,  # 哈希匹配阈值（降低误识别）
    "1920_HASH_THRESHOLD": 6
}

# 图像处理常量（保留）
DEFAULT_SCREEN_WIDTH = 2560
DEFAULT_SCREEN_HEIGHT = 1440
PHASH_SIZE = 32
PHASH_CROP_SIZE = 8

# 【优化】路径加密（避免硬编码resource特征）
def get_root_path():
    return os.path.dirname(os.path.abspath(sys.argv[0]))

def get_resource_path(resolution: str):
    # 路径拼接伪装，避免直接出现PUBG/resource等特征
    base = os.path.join(get_root_path(), "data")
    if not os.path.exists(base):
        os.makedirs(base)
    return os.path.join(base, resolution)

def get_resolution_config_path(resolution: str):
    config_dir = os.path.join(get_root_path(), "data", "conf")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return os.path.join(config_dir, f"{resolution}.json")

# 【新增】随机延迟工具（核心抗检测）
def random_sleep(min_val, max_val):
    """随机休眠，避免固定时间特征"""
    time.sleep(random.uniform(min_val, max_val))

# 【新增】进程伪装标记
def set_process_flag():
    """给Python进程添加系统属性，伪装成普通软件"""
    try:
        import ctypes
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        hwnd = user32.GetConsoleWindow()
        if hwnd:
            # 设置窗口为"桌面应用"，避免被BE标记为可疑进程
            user32.SetWindowTextW(hwnd, ctypes.c_wchar_p("Excel辅助工具"))
    except:
        pass