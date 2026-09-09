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
            "旧 ExportEngine 导出接口已禁用；请通过主窗口导出流程生成带 manifest 的科研文件"
        )

    def export_auto_path_to_file(self, save_path: str):
        """导出自动路径到文件。
        直接使用已有的 path_points/path_status（已由 _action_plan 同步）；
        物理直线/转弯预览由主窗口的路径导出流程负责，旧接口本身继续禁用，
        不再破坏性覆写，保留段间过渡状态。
        """
        self.export_to_file(save_path)
