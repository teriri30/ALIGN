# -*- coding: utf-8 -*-
"""按会话记录用户操作的追加式日志（问题复现与标定闭环的审计底座）。

为什么需要它
------------
``project_log.ProjectJournal`` 只记"错误 / 配置变更"这类结论，回答的是
"这次为什么失败"。但用户报的问题经常是"我做了一串操作，结果不对"——没有
交互序列就无法复现，也看不出他在哪一步改了参数、哪一步拖动过作业线。
本模块记录**每一次有意义的用户动作**（菜单、任务行、参数、标定拖动、导出），
跟"有没有保存项目"无关，一行一个 JSON，追加写入。

与 ProjectJournal 的分工
------------------------
- ``ProjectJournal``：项目日志，面向"这次失败的原因"，进日志面板；
- ``ActionLog``：交互日志，面向"用户到底做了什么"，是标定数据的来源与
  人工纠偏的审计记录，不进日志面板（避免刷屏）。

设计约束
--------
- **绝不打断界面**：任何写盘异常都在内部吞掉，返回值只用于测试；
- **追加式**（append-only）+ 按大小轮转，不重写历史；
- 每次记录都带上下文（应用版本 / 影像 / 项目），日志可脱离会话单独阅读；
- 字段名稳定，离线分析按 ``action`` 前缀过滤，例如 ``calibration.``。

日志位置：``%LOCALAPPDATA%\\ALIGN\\logs\\actions\\actions-<会话>.jsonl``
（测试与便携场景可用 ``ALIGN_ACTION_LOG_DIR`` 覆盖；``ALIGN_DISABLE_ACTION_LOG=1``
可整体停用）。
"""
from __future__ import annotations

import datetime
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # 与运行期路径保持一致；测试里 runtime_paths 可用
    from runtime_paths import logs_dir as _logs_dir
except Exception:  # pragma: no cover - 仅在极度受限的环境下发生
    def _logs_dir() -> Path:  # type: ignore
        return Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "ALIGN" / "logs"

MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
SCHEMA = "align-action-log-v1"
_SESSION = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def actions_dir() -> Path:
    override = str(os.environ.get("ALIGN_ACTION_LOG_DIR") or "").strip()
    if override:
        return Path(override)
    return Path(_logs_dir()) / "actions"


def _disabled() -> bool:
    return str(os.environ.get("ALIGN_DISABLE_ACTION_LOG") or "").strip() in {"1", "true", "True"}


def _jsonable(value: Any) -> Any:
    """只保留能进 JSON 的类型，其余转字符串（日志不该因为一个对象就写失败）。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


class ActionLog:
    """进程内单例：会话 id 固定，写入串行化。"""

    _instance: Optional["ActionLog"] = None
    _lock = threading.RLock()

    def __new__(cls) -> "ActionLog":
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._context: Dict[str, Any] = {}
            instance._seq = 0
            instance._session = _SESSION
            instance._path_override = ""
            instance.last_error = ""
            cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------ 基础
    @property
    def session(self) -> str:
        return self._session

    def configure(self, **context: Any) -> None:
        """更新上下文（后续每条记录都会带上）；None 值表示清除该键。"""
        with self._lock:
            for key, value in context.items():
                if value is None:
                    self._context.pop(str(key), None)
                else:
                    self._context[str(key)] = _jsonable(value)

    def context(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._context)

    def log_path(self) -> Path:
        if self._path_override:
            return Path(self._path_override)
        return actions_dir() / f"actions-{self._session}.jsonl"

    def set_log_path(self, path: str) -> None:
        """测试用：把日志固定写到指定文件。"""
        with self._lock:
            self._path_override = str(path)

    # ------------------------------------------------------------------ 写入
    def record(
        self,
        action: str,
        target: str = "",
        *,
        level: str = "INFO",
        **payload: Any,
    ) -> Dict[str, Any]:
        now = datetime.datetime.now().astimezone()
        with self._lock:
            self._seq += 1
            record: Dict[str, Any] = {
                "schema": SCHEMA,
                "ts": now.isoformat(timespec="milliseconds"),
                "session": self._session,
                "seq": int(self._seq),
                "level": str(level or "INFO").upper(),
                "action": str(action or ""),
                "target": str(target or ""),
                "context": dict(self._context),
            }
            if payload:
                record["payload"] = _jsonable(payload)
            if _disabled():
                return record
            try:
                path = self.log_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed(path)
                with open(path, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.last_error = ""
            except Exception as exc:  # 日志失败不得影响主流程
                self.last_error = f"{type(exc).__name__}: {exc}"
            return record

    @staticmethod
    def _rotate_if_needed(path: Path) -> None:
        try:
            if not path.exists() or path.stat().st_size < MAX_BYTES:
                return
            oldest = path.with_name(path.name + f".{BACKUP_COUNT}")
            if oldest.exists():
                oldest.unlink()
            for index in range(BACKUP_COUNT - 1, 0, -1):
                source = path.with_name(path.name + f".{index}")
                if source.exists():
                    os.replace(source, path.with_name(path.name + f".{index + 1}"))
            os.replace(path, path.with_name(path.name + ".1"))
        except OSError:
            return

    # ------------------------------------------------------------------ 读取
    def read(self, path: Optional[str] = None, limit: int = 0) -> List[Dict[str, Any]]:
        target = Path(path) if path else self.log_path()
        if not target.is_file():
            return []
        records: List[Dict[str, Any]] = []
        try:
            with open(target, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(item, dict):
                        records.append(item)
        except OSError:
            return []
        if limit and limit > 0:
            return records[-limit:]
        return records

    def recent(self, limit: int = 200, action_prefix: str = "") -> List[Dict[str, Any]]:
        records = self.read(limit=0)
        if action_prefix:
            records = [
                item for item in records
                if str(item.get("action") or "").startswith(action_prefix)
            ]
        return records[-max(1, int(limit)):]

    def export_to(self, directory: str) -> List[str]:
        """把当前会话的全部日志（含轮转文件）复制到目录，返回落盘路径。"""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        written: List[str] = []
        path = self.log_path()
        candidates: Iterable[Path] = [path] + [
            path.with_name(path.name + f".{index}")
            for index in range(1, BACKUP_COUNT + 1)
        ]
        for source in candidates:
            if not source.is_file():
                continue
            destination = target_dir / source.name
            try:
                destination.write_bytes(source.read_bytes())
                written.append(str(destination))
            except OSError:
                continue
        return written


def format_records(records: Iterable[Dict[str, Any]], *, max_payload_chars: int = 200) -> str:
    """把操作记录排成可读文本：一行一个动作，参数压成 key=value。

    给“操作记录”窗口和导出的 txt 用；不改变 JSONL 原文，日志仍是逐行 JSON。
    """
    lines: List[str] = []
    for item in records or []:
        if not isinstance(item, dict):
            continue
        timestamp = str(item.get("ts") or "")
        action = str(item.get("action") or "")
        target = str(item.get("target") or "")
        payload = item.get("payload") or {}
        detail = ""
        if isinstance(payload, dict) and payload:
            detail = "  " + "  ".join(
                "%s=%s" % (key, value) for key, value in payload.items()
                if value not in (None, "", [], {})
            )
            if len(detail) > max_payload_chars:
                detail = detail[:max_payload_chars].rstrip() + " ..."
        line = "%s  [%s]" % (timestamp, action)
        if target:
            line += "  " + target
        lines.append(line + detail)
    return "\n".join(lines)


def action_log() -> ActionLog:
    return ActionLog()


def record(action: str, target: str = "", **payload: Any) -> Dict[str, Any]:
    """模块级快捷方式：``record("menu.action", "运行识别", shortcut="Ctrl+R")``。"""
    return action_log().record(action, target, **payload)


def configure(**context: Any) -> None:
    action_log().configure(**context)
