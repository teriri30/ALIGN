"""导航文件导出"""
from typing import List, Tuple
from config import Config, AppLogger
from state import AppState, AutoPathSegment
from geo import GeoUtils


class ExportEngine:
    """Deprecated unsafe export API retained only to fail closed for callers."""
    def __init__(self, geo: GeoUtils, state: AppState):
        self.geo = geo
        self.state = state
        self.cfg = Config()
        self.log = AppLogger()

    def export_to_file(self, save_path: str):
        raise RuntimeError(
            "旧 ExportEngine 导出接口已禁用；必须通过主窗口的正式导出门禁和溯源清单导出"
        )

    def export_auto_path_to_file(self, save_path: str):
        """导出自动路径到文件。
        直接使用已有的 path_points/path_status（已由 _action_plan 同步），
        不再破坏性覆写，保留段间过渡状态。
        """
        self.export_to_file(save_path)
