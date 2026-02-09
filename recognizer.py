import os
import json
import ctypes
import threading
import time
import numpy as np
import cv2
from recognize_object import RecognizeObject
from cv_utils import CVUtils
from config import (
    get_resource_path, get_resolution_config_path,
    DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT,
    ANTI_DETECT, set_process_flag
)

# 引入UI和Lua生成器（容错处理，无特征）
try:
    from luaScriptGenerator import luaScriptGenerator
except ImportError as e:
    print(f"警告：模块缺失 - {e}")
    class luaScriptGenerator:
        @staticmethod
        def generate_lua_file(*args, **kwargs):
            pass

class Recognizer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.__init_once()
                set_process_flag()  # 初始化时执行进程伪装
        return cls._instance

    def __init_once(self):
        if hasattr(self, 'init_success'):
            return
        self.init_success = False
        self.bps = []
        self.poses = []
        self.weapons = []
        self.attachment1s = []
        self.attachment2s = []

        # 分辨率与坐标
        self.screen_width = DEFAULT_SCREEN_WIDTH
        self.screen_height = DEFAULT_SCREEN_HEIGHT
        self._init_coords()

        # 运行状态（跟随抗检测配置）
        self.fps = ANTI_DETECT["RECOG_FPS"]
        self.RecogFPS = self.fps
        self.isrunning = False
        self.pthread = None
        self.scope_mode = 0
        self.blood_enable = False
        self.bag1 = False
        self.bag2 = False
        self.bag3 = False

        # 识别结果
        self.weapon1 = "none"
        self.w1a1 = 0
        self.w1a2 = 0
        self.weapon2 = "none"
        self.w2a1 = 0
        self.w2a2 = 0
        self.pose = 0
        # Lua输出路径（伪装为配置文件）
        # self.lua_path = os.path.join(get_resource_path(""), "conf.lua")

        # 初始化
        self._initialize()

    def _init_coords(self):
        """初始化坐标（无硬编码特征）"""
        coords = [
            "bp1_x", "bp1_y", "bp1_width", "bp1_height",
            "bp2_x", "bp2_y", "bp2_width", "bp2_height",
            "bp3_x", "bp3_y", "bp3_width", "bp3_height",
            "weapon1_x", "weapon1_y", "weapon1_width", "weapon1_height",
            "w1a1_x", "w1a1_y", "w1a1_width", "w1a1_height",
            "w1a2_x", "w1a2_y", "w1a2_width", "w1a2_height",
            "w2a1_x", "w2a1_y", "w2a1_width", "w2a1_height",
            "w2a2_x", "w2a2_y", "w2a2_width", "w2a2_height",
            "w1_x", "w1_y", "w1_width", "w1_height"
        ]
        for coord in coords:
            setattr(self, coord, 0)

    def _initialize(self):
        """优化初始化 - 低频磁盘IO"""
        try:
            # 调用Windows用户32位库
            user32 = ctypes.windll.user32
            if user32 is not None:
                # 获取屏幕宽（像素）
                self.screen_width = user32.GetSystemMetrics(0)
                # 获取屏幕高（像素）
                self.screen_height = user32.GetSystemMetrics(1)
            self.resolution_str = f"{self.screen_width}_{self.screen_height}"

            # 检查资源（伪装为data目录）
            if not os.path.exists(get_resource_path(self.resolution_str)):
                raise Exception(f"不支持分辨率 {self.resolution_str}")

            # 加载配置
            json_path = get_resolution_config_path(self.resolution_str)
            if not os.path.exists(json_path):
                raise Exception(f"配置文件缺失 {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                self._parse_json_config(json.load(f))

            # 加载模板（单次加载，避免重复磁盘IO）
            self._load_all_templates()

            self.init_success = True
            print(f"初始化成功 | 分辨率: {self.resolution_str} | 帧率: {self.fps}FPS")
        except Exception as e:
            print(f"初始化失败: {e}")
            self.init_success = False

    def _parse_json_config(self, json_data: dict):
        """解析配置（保留）"""
        required_keys = ["weapon1", "w1","weapon2", "w2", "bp1", "bp2", "bp3", "pose", "w1scope", "w1a1", "w1a2", "w2scope", "w2a1", "w2a2"]
        for key in required_keys:
            if key not in json_data or len(json_data[key]) < 4:
                raise Exception(f"配置缺失 {key}")
            setattr(self, f"{key}_x", json_data[key][0])
            setattr(self, f"{key}_y", json_data[key][1])
            setattr(self, f"{key}_width", json_data[key][2])
            setattr(self, f"{key}_height", json_data[key][3])

    def _load_all_templates(self):
        """优化模板加载 - 降低IO频率"""
        self._load_bag_templates()
        self._load_poses_templates()
        self._load_weapon_templates()
        self._load_attachment_templates()

    def _load_poses_templates(self):
        pose_dir = os.path.join(get_resource_path(self.resolution_str), "pose")
        if not os.path.exists(pose_dir):
            return

        for filename in os.listdir(pose_dir):
            if filename.endswith(".png"):
                pose_name = os.path.splitext(filename)[0]
                img = cv2.imread(os.path.join(pose_dir, filename), cv2.IMREAD_COLOR)
                if img is not None:
                    self.poses.append(RecognizeObject(pose_name, img, 0))
    def _load_bag_templates(self):
        for idx in [1, 2, 3]:
            img_path = os.path.join(get_resource_path(self.resolution_str), "bg", f"bg{idx}.png")
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    self.bps.append(RecognizeObject(f"bg{idx}", img, idx))

    def _load_weapon_templates(self):
        gun_dir = os.path.join(get_resource_path(self.resolution_str), "icon")
        if not os.path.exists(gun_dir):
            return
        for filename in os.listdir(gun_dir):
            if filename.endswith(".png"):
                gun_name = os.path.splitext(filename)[0]
                img = cv2.imread(os.path.join(gun_dir, filename), cv2.IMREAD_COLOR)
                if img is not None:
                    self.weapons.append(RecognizeObject(gun_name, img, 0))

    def _load_attachment_templates(self):
        self.attachment1s.clear()
        self.attachment2s.clear()
        att1_dir = os.path.join(get_resource_path(self.resolution_str), "part", "1")
        att2_dir = os.path.join(get_resource_path(self.resolution_str), "part", "2")
        self._load_attachments(att1_dir, self.attachment1s)
        self._load_attachments(att2_dir, self.attachment2s)

    def _load_attachments(self, dir_path: str, target_list: list):
        if not os.path.exists(dir_path):
            return
        for filename in os.listdir(dir_path):
            if filename.endswith(".png"):
                attach_name = os.path.splitext(filename)[0]
                img = cv2.imread(os.path.join(dir_path, filename), cv2.IMREAD_COLOR)
                if img is not None:
                    id_val = int(attach_name.split("_")[0]) if "_" in attach_name else (int(attach_name) if attach_name.isdigit() else 0)
                    target_list.append(RecognizeObject(attach_name, img, id_val))
    def weapon_name_recognize(self, screen: np.ndarray, x: int, y: int, width: int, height: int) -> int:
        """优化识别 - 提前判断模板是否存在"""
        if not self.weapons or screen is None:
            return -1
        crop = CVUtils.crop_mat(screen, x, y, width, height)
        if crop is None:
            return -1
        crop = CVUtils.to_black_white_above200(crop, False)
        hash_val = CVUtils.pHash(crop)
        min_dist = ANTI_DETECT["HASH_THRESHOLD"]
        windex = -1
        for i, weapon in enumerate(self.weapons):
            dist = CVUtils.HammingDistance(weapon.phash, hash_val)
            if dist < min_dist:
                min_dist = dist
                windex = i
        return windex

    def is_bag_open(self, screen: np.ndarray) -> bool:
        """优化背包判断 - 动态阈值"""
        if screen is None or len(self.bps) < 2:
            # print(len(self.bps))
            return False
        bp1_crop = CVUtils.crop_mat(screen, self.bp1_x, self.bp1_y, self.bp1_width, self.bp1_height)
        bp2_crop = CVUtils.crop_mat(screen, self.bp2_x, self.bp2_y, self.bp2_width, self.bp2_height)
        bp3_crop = CVUtils.crop_mat(screen, self.bp3_x, self.bp3_y, self.bp3_width, self.bp3_height)
        if bp1_crop is None or bp2_crop is None or bp3_crop is None:
            return False

        # 动态阈值（跟随分辨率）
        threshold = ANTI_DETECT["1920_HASH_THRESHOLD"] if self.screen_width == 1920 else ANTI_DETECT["HASH_THRESHOLD"]
        self.bag1 = CVUtils.HammingDistance(CVUtils.pHash(bp1_crop), self.bps[0].phash) < threshold
        self.bag2 = CVUtils.HammingDistance(CVUtils.pHash(bp2_crop), self.bps[1].phash) < threshold
        self.bag3 = CVUtils.HammingDistance(CVUtils.pHash(bp3_crop), self.bps[2].phash) < threshold

        return self.bag1 or self.bag2 or self.bag3

    def attach_recognize(self, screen: np.ndarray, lua_tk_inst=None, lua_gen_inst=None):
        """配件识别 - 传递UI/Lua实例"""
        # self._recognize_single_weapon_attach(screen, "w1")
        self._recognize_single_weapon_attach(screen, "w2")

    def _recognize_single_weapon_attach(self, screen: np.ndarray, weapon_prefix: str):
        """简化配件识别 - 降低计算量"""
        a1_x = getattr(self, f"{weapon_prefix}a1_x")
        a1_y = getattr(self, f"{weapon_prefix}a1_y")
        a1_w = getattr(self, f"{weapon_prefix}a1_width")
        a1_h = getattr(self, f"{weapon_prefix}a1_height")
        a2_x = getattr(self, f"{weapon_prefix}a2_x")
        a2_y = getattr(self, f"{weapon_prefix}a2_y")
        a2_w = getattr(self, f"{weapon_prefix}a2_width")
        a2_h = getattr(self, f"{weapon_prefix}a2_height")

        a1_crop = CVUtils.crop_mat(screen, a1_x, a1_y, a1_w, a1_h)
        a2_crop = CVUtils.crop_mat(screen, a2_x, a2_y, a2_w, a2_h)
        if a1_crop is not None:
            a1_crop = CVUtils.to_black_white1(a1_crop, False)
            self._match_attachment(a1_crop, self.attachment1s, weapon_prefix, "a1")
        if a2_crop is not None:
            a2_crop = CVUtils.to_black_white1(a2_crop, False)
            self._match_attachment(a2_crop, self.attachment2s, weapon_prefix, "a2")

    def _match_attachment(self, crop: np.ndarray, attach_list: list, weapon_prefix: str, attach_suffix: str):
        """简化配件匹配"""
        if not attach_list:
            return
        hash_val = CVUtils.pHash(crop)
        min_dist = 20
        best_id = 0
        for attach in attach_list:
            dist = CVUtils.HammingDistance(hash_val, attach.phash)
            if dist < min_dist:
                min_dist = dist
                best_id = attach.id
        setattr(self, f"{weapon_prefix}{attach_suffix}", best_id)

    def test_isbag(self, screen: np.ndarray):
        """测试是否背包"""
        if screen is None or len(self.bps) < 2:
            print(len(self.bps))
            return False
        bp2_crop = CVUtils.crop_mat(screen, self.bp2_x, self.bp2_y, self.bp2_width, self.bp2_height)
        bp3_crop = CVUtils.crop_mat(screen, self.bp3_x, self.bp3_y, self.bp3_width, self.bp3_height)
        if bp3_crop is None:
            return False

        # # 确保保存目录存在（不存在则创建）
        save_dir = "E:\pyProject\pubg_ps\crop_output"
        bp2_save_path = os.path.join(save_dir, "bp222_crop.png")
        bp1_saved = cv2.imwrite(bp2_save_path, bp2_crop)
        print(bp1_saved)

        threshold = ANTI_DETECT["1920_HASH_THRESHOLD"] if self.screen_width == 1920 else ANTI_DETECT["HASH_THRESHOLD"]
        # print(self.bps)
        self.bag3 = CVUtils.HammingDistance(CVUtils.pHash(bp3_crop), self.bps[2].phash) < threshold
        print(self.bag3)
    def test_wwws(self, screen: np.ndarray):
        """测试1号和2号枪"""
        if screen is None:
            return
        """执行核心识别逻辑"""
        if self.is_bag_open(screen):
            if self.bag2:
                w2_idx = self.weapon_name_recognize(screen, self.w2_x, self.w2_y, self.w2_width, self.w2_height)
                if w2_idx != -1:
                    self.weapon2 = self.weapons[w2_idx].name

        print(self.weapon2)

    def pose_recognize(self, screen):
        """姿态识别"""
        pose_mat = CVUtils.crop_mat(screen, self.pose_x, self.pose_y, self.pose_width, self.pose_height)

        if pose_mat is None:
            return -1
        crop = CVUtils.to_black_white_above200(pose_mat, False)
        hash_val = CVUtils.pHash(crop)
        min_dist = ANTI_DETECT["HASH_THRESHOLD"]
        windex = -1
        for i, pose in enumerate(self.poses):
            dist = CVUtils.HammingDistance(pose.phash, hash_val)
            if dist < min_dist:
                min_dist = dist
                windex = i
        return windex

    def weapon_choose(self, screen):
        """枪械选择（判断当前手持枪械）"""
        w1 = CVUtils.crop_mat(screen, self.weapon1_x, self.weapon1_y, self.weapon1_width, self.weapon1_height)
        w2 = CVUtils.crop_mat(screen, self.weapon2_x, self.weapon2_y, self.weapon2_width, self.weapon2_height)
        w1 = CVUtils.to_black_white_avg(w1)
        w2 = CVUtils.to_black_white_avg(w2)
        # 统计白色像素数
        n1 = np.sum((w1 == 255).all(axis=2))
        n2 = np.sum((w2 == 255).all(axis=2))
        if n1 - n2 > 100:
            return 1
        elif n2 - n1 > 100:
            return 2
        else:
            return 0
    def do_recognize(self, screen: np.ndarray):
        if screen is None:
            return
        """执行核心识别逻辑"""
        if self.is_bag_open(screen):
            # if self.bag1:
            #     w1_idx = self.weapon_name_recognize(screen, self.w1_x, self.w1_y, self.w1_width, self.w1_height)
            #     if w1_idx != -1:
            #         self.weapon1 = self.weapons[w1_idx].name
            if self.bag2:
                w2_idx = self.weapon_name_recognize(screen, self.w2_x, self.w2_y, self.w2_width, self.w2_height)
                if w2_idx != -1:
                    self.weapon2 = self.weapons[w2_idx].name

            self.attach_recognize(screen)
        else:
            self.bag1 = False
            self.bag2 = False
            self.bag3 = False

    def start(self):
        self.isrunning = True

    def stop(self):
        self.isrunning = False