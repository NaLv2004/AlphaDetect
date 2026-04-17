from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CFGBlock:
    id: str
    preds: list[str] = field(default_factory=list)
    succs: list[str] = field(default_factory=list)


def link_blocks(blocks: dict[str, CFGBlock], src: str, dst: str) -> None:
    if dst not in blocks[src].succs:
        blocks[src].succs.append(dst)
    if src not in blocks[dst].preds:
        blocks[dst].preds.append(src)

