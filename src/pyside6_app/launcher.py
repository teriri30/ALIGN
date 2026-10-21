"""
launcher.py - 项目管理器
"""
import os, json
from builtin_test_assets import builtin_test_image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QFileDialog, QMessageBox)
from pyside6_app.styles import COLORS

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECENT_FILE = os.path.join(APP_ROOT, "recent_files.json")
def load_recent():
    try:
        if os.path.exists(RECENT_FILE):
            with open(RECENT_FILE,"r",encoding="utf-8") as f: return json.load(f)
    except Exception: pass
    return []
def save_recent(recent):
    try:
        with open(RECENT_FILE,"w",encoding="utf-8") as f: json.dump(recent[:20],f,ensure_ascii=False,indent=2)
    except Exception: pass

def project_entries():
    """Return the stable built-in test image followed by valid recent files."""
    entries = []
    test_image = builtin_test_image()
    if test_image:
        entries.append({"kind": "builtin_test", "label": "内置测试影像", "path": test_image})
    seen = {os.path.normcase(os.path.abspath(item["path"])) for item in entries}
    for path in load_recent():
        if not os.path.isfile(path):
            continue
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized in seen:
            continue
        seen.add(normalized)
        entries.append({"kind": "recent", "label": "最近项目", "path": path})
    return entries

class LauncherDialog(QDialog):
    file_selected = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("\u667a\u80fd\u519c\u673a\u89c4\u5212\u7cfb\u7edf")
        self.setFixedSize(600,480); self.setAcceptDrops(True)
        self.selected_path = ""
        self._recent = load_recent(); self._build_ui()
    def _build_ui(self):
        lo = QVBoxLayout(self); lo.setSpacing(12); lo.setContentsMargins(24,24,24,20)
        t = QLabel("\u667a\u80fd\u519c\u673a\u89c4\u5212\u7cfb\u7edf")
        t.setStyleSheet(f"font-size:22px;font-weight:600;color:{COLORS['text_bright']};")
        t.setAlignment(Qt.AlignCenter); lo.addWidget(t)
        st = QLabel("\u9009\u62e9\u6216\u5bfc\u5165\u6b63\u5c04\u5f71\u50cf (TIF/TIFF) \u5f00\u59cb\u5904\u7406")
        st.setStyleSheet(f"font-size:12px;color:{COLORS['text_dim']};")
        st.setAlignment(Qt.AlignCenter); lo.addWidget(st)
        dp = QLabel("\u5c06 TIF \u6587\u4ef6\u62d6\u653e\u5230\u8fd9\u91cc\n\u6216\u70b9\u51fb\u4e0b\u65b9\u6309\u94ae\u5bfc\u5165")
        dp.setAlignment(Qt.AlignCenter); dp.setMinimumHeight(80)
        dp.setStyleSheet(f"QLabel{{background:{COLORS['bg_darkest']};border:2px dashed {COLORS['border_light']};border-radius:10px;color:{COLORS['text_dim']};font-size:13px;padding:20px;}}QLabel:hover{{border-color:{COLORS['accent']};background:{COLORS['bg_accent']};}}")
        lo.addWidget(dp)
        bl = QHBoxLayout()
        bi = QPushButton("  \u5bfc\u5165 TIF \u6587\u4ef6  ")
        bi.setObjectName("primary"); bi.setFixedWidth(200); bi.setMinimumHeight(40)
        bi.clicked.connect(self._on_import)
        bl.addStretch(); bl.addWidget(bi); bl.addStretch(); lo.addLayout(bl)
        lr = QLabel("内置测试影像与最近项目")
        lr.setStyleSheet(f"font-size:12px;color:{COLORS['text_dim']};font-weight:500;")
        lo.addWidget(lr)
        self.list_widget = QListWidget(); self._refresh_list()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        lo.addWidget(self.list_widget)
        bt = QHBoxLayout()
        bd = QPushButton("\u5220\u9664"); bd.clicked.connect(self._on_delete)
        bc = QPushButton("\u53d6\u6d88"); bc.clicked.connect(self.reject)
        bo = QPushButton("\u6253\u5f00\u9879\u76ee")
        bo.clicked.connect(self._on_open_selected)
        bt.addWidget(bd); bt.addStretch(); bt.addWidget(bc); bt.addWidget(bo)
        lo.addLayout(bt)
    def _refresh_list(self):
        self.list_widget.clear()
        for entry in project_entries():
            p = entry["path"]
            prefix = "[内置测试]" if entry["kind"] == "builtin_test" else "[最近]"
            i = QListWidgetItem(f"{prefix} {os.path.basename(p)}  ({p})")
            i.setData(Qt.UserRole,p)
            i.setData(Qt.UserRole + 1, entry["kind"])
            self.list_widget.addItem(i)
    def _on_import(self):
        paths,_ = QFileDialog.getOpenFileNames(self,"\u9009\u62e9\u6b63\u5c04\u5f71\u50cf","","GeoTIFF (*.tif *.tiff);;\u6240\u6709\u6587\u4ef6 (*)")
        if paths:
            for p in paths:
                if p not in self._recent: self._recent.insert(0,p)
            save_recent(self._recent)
            self.selected_path = paths[0]
            self._refresh_list(); self.file_selected.emit(paths[0]); self.accept()
    def _on_delete(self):
        i = self.list_widget.currentItem()
        if i and i.data(Qt.UserRole + 1) != "builtin_test":
            p = i.data(Qt.UserRole)
            if p in self._recent: self._recent.remove(p); save_recent(self._recent); self._refresh_list()
    def _on_open_selected(self):
        i = self.list_widget.currentItem()
        if i:
            p = i.data(Qt.UserRole)
            if os.path.exists(p):
                if p in self._recent: self._recent.remove(p)
                self._recent.insert(0,p); save_recent(self._recent)
                self.selected_path = p
                self.file_selected.emit(p); self.accept()
            else: QMessageBox.warning(self,"\u6587\u4ef6\u4e0d\u5b58\u5728",f"\u6587\u4ef6\u5df2\u4e0d\u5b58\u5728:\n{p}")
    def _on_item_double_clicked(self,i): self._on_open_selected()
    def dragEnterEvent(self,e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self,e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if p.lower().endswith((".tif",".tiff")):
                if p not in self._recent: self._recent.insert(0,p)
                self.selected_path = p
                save_recent(self._recent); self.file_selected.emit(p); self.accept(); return
