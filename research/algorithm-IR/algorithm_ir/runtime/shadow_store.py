from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShadowStore:
    object_versions: dict[int, list[str]] = field(default_factory=dict)
    field_writers: dict[tuple[int, str], str] = field(default_factory=dict)
    item_writers: dict[tuple[int, object], str] = field(default_factory=dict)
    container_membership: dict[int, set[str]] = field(default_factory=dict)

    def note_value(self, py_obj, rid: str) -> None:
        if py_obj is None:
            return
        obj_id = id(py_obj)
        self.object_versions.setdefault(obj_id, []).append(rid)
        if isinstance(py_obj, (list, dict, set, tuple)):
            self.container_membership.setdefault(obj_id, set())

    def note_append(self, container, member_rid: str, writer_event: str) -> None:
        obj_id = id(container)
        self.container_membership.setdefault(obj_id, set()).add(member_rid)
        self.object_versions.setdefault(obj_id, []).append(writer_event)

    def note_set_item(self, container, key, writer_event: str) -> None:
        obj_id = id(container)
        self.item_writers[(obj_id, self._freeze_key(key))] = writer_event
        self.object_versions.setdefault(obj_id, []).append(writer_event)

    def note_set_attr(self, owner, attr: str, writer_event: str) -> None:
        obj_id = id(owner)
        self.field_writers[(obj_id, attr)] = writer_event
        self.object_versions.setdefault(obj_id, []).append(writer_event)

    def _freeze_key(self, key):
        if isinstance(key, list):
            return tuple(key)
        if isinstance(key, dict):
            return tuple(sorted(key.items()))
        return key

