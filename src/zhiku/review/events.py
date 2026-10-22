"""JSONL 事件日志：按 rule_id + date 幂等去重。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from zhiku.models import TriggerEvent


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> list[TriggerEvent]:
        if not self.path.is_file():
            return []
        events: list[TriggerEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            events.append(TriggerEvent.model_validate_json(text))
        return events

    def _keys(self) -> set[tuple[str, date]]:
        return {(e.rule_id, e.date) for e in self.read()}

    def append(self, event: TriggerEvent) -> bool:
        """写入一条事件。若 rule_id+date 已存在则跳过，返回是否真正写入。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if (event.rule_id, event.date) in self._keys():
            return False
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")
        return True

    def append_many(self, events: list[TriggerEvent]) -> int:
        written = 0
        for event in events:
            if self.append(event):
                written += 1
        return written

    def in_range(self, start: date, end: date) -> list[TriggerEvent]:
        return [e for e in self.read() if start <= e.date <= end]
