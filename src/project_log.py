"""Persistent project journal for errors, operations, and configuration changes."""

import datetime
import json
import os
import threading
import traceback
from typing import Dict, List, Optional
from runtime_paths import logs_dir


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# DECISION-018: logs survive replacement of the frozen program directory.
LOG_DIR = str(logs_dir())
JOURNAL_PATH = os.path.join(LOG_DIR, "project_journal.jsonl")
JOURNAL_MAX_BYTES = 10 * 1024 * 1024
JOURNAL_BACKUP_COUNT = 3


class ProjectJournal:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._lock = threading.RLock()
            cls._instance._thread_hook_installed = False
        return cls._instance

    def write(
        self,
        level: str,
        event_type: str,
        message: str,
        module: str = "",
        details: Optional[Dict] = None,
        traceback_text: str = "",
    ) -> Dict:
        os.makedirs(LOG_DIR, exist_ok=True)
        record = {
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "level": str(level).upper(),
            "event_type": str(event_type),
            "module": str(module),
            "message": str(message),
        }
        if details:
            record["details"] = details
        if traceback_text:
            record["traceback"] = traceback_text
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._rotate_if_needed()
            with open(JOURNAL_PATH, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return record

    @staticmethod
    def _rotate_if_needed():
        try:
            if not os.path.exists(JOURNAL_PATH):
                return
            if os.path.getsize(JOURNAL_PATH) < JOURNAL_MAX_BYTES:
                return
            oldest = f"{JOURNAL_PATH}.{JOURNAL_BACKUP_COUNT}"
            if os.path.exists(oldest):
                os.remove(oldest)
            for index in range(JOURNAL_BACKUP_COUNT - 1, 0, -1):
                source = f"{JOURNAL_PATH}.{index}"
                if os.path.exists(source):
                    os.replace(source, f"{JOURNAL_PATH}.{index + 1}")
            os.replace(JOURNAL_PATH, f"{JOURNAL_PATH}.1")
        except OSError:
            # Journal rotation failure must not corrupt the active operation.
            return

    def info(self, message: str, module: str = "", details: Optional[Dict] = None):
        return self.write("INFO", "operation", message, module, details)

    def change(self, message: str, module: str = "", details: Optional[Dict] = None):
        return self.write("INFO", "change", message, module, details)

    def error(
        self,
        message: str,
        module: str = "",
        details: Optional[Dict] = None,
        traceback_text: str = "",
    ):
        return self.write(
            "ERROR",
            "error",
            message,
            module,
            details,
            traceback_text,
        )

    def recent(self, limit: int = 500, event_type: str = "") -> List[Dict]:
        if not os.path.exists(JOURNAL_PATH):
            return []
        with self._lock:
            with open(JOURNAL_PATH, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
        records = []
        for line in lines[-max(limit * 3, limit):]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event_type and record.get("event_type") != event_type:
                continue
            records.append(record)
        return records[-limit:]

    def install_thread_exception_hook(self):
        if self._thread_hook_installed or not hasattr(threading, "excepthook"):
            return
        previous = threading.excepthook

        def hook(args):
            tb_text = "".join(
                traceback.format_exception(
                    args.exc_type,
                    args.exc_value,
                    args.exc_traceback,
                )
            )
            self.error(
                f"Unhandled thread exception: {args.exc_type.__name__}: {args.exc_value}",
                module=getattr(args.thread, "name", "thread"),
                traceback_text=tb_text,
            )
            previous(args)

        threading.excepthook = hook
        self._thread_hook_installed = True


def format_record(record: Dict) -> str:
    timestamp = record.get("timestamp", "")
    level = record.get("level", "")
    event_type = record.get("event_type", "")
    module = record.get("module", "")
    message = record.get("message", "")
    header = f"{timestamp} [{level}] [{event_type}]"
    if module:
        header += f" [{module}]"
    lines = [f"{header} {message}"]
    details = record.get("details")
    if details:
        lines.append(json.dumps(details, ensure_ascii=False, indent=2, default=str))
    if record.get("traceback"):
        lines.append(record["traceback"].rstrip())
    return "\n".join(lines)
