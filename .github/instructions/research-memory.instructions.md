---
description: "Use when reading or modifying research memory files (state.json, experiment-log.md, idea-bank.md, etc.). Ensures consistent format, proper timestamps, and append-only updates."
applyTo: "research/memory/**"
---

# Research Memory File Conventions

## General Rules

1. **Append, don't overwrite**: Add new entries at the end (or update status of existing entries). Never delete history.
2. **Timestamp everything**: Use ISO 8601 format `YYYY-MM-DD HH:MM` for all entries.
3. **Cross-reference**: Link entries across files (e.g., experiment results → idea, decision → literature).

## state.json Format

```json
{
  "last_updated": "2026-03-31 10:00",
  "active_topics": [
    {
      "name": "topic-name",
      "phase": "literature-search|ideation|coding|experiment|writing|review",
      "status": "Current status description",
      "next_action": "Next planned action",
      "folder": "research/topic-name/"
    }
  ],
  "global_notes": "Cross-topic observations and high-level strategy notes"
}
```

## experiment-log.md Format

```markdown
## [YYYY-MM-DD] Experiment Title
- **Topic**: research/<topic>/
- **Parameters**: N=..., K=..., SNR=..., etc.
- **Result**: BER=..., FER=..., Time=...
- **Observation**: Key finding
- **Files**: path/to/results
```

## idea-bank.md Format

```markdown
## [ID-XXX] Idea Title
- **Status**: proposed|exploring|coding|testing|writing|completed|abandoned
- **Created**: YYYY-MM-DD
- **Updated**: YYYY-MM-DD
- **Summary**: One-sentence description
- **Related**: Links to literature, experiments, decisions
```

## decision-history.md Format

```markdown
## [YYYY-MM-DD] Decision Title
- **Decision**: What was decided
- **Rationale**: Why
- **Alternatives**: What was considered
- **Impact**: Expected effect
```

## experience-base.md Format

```markdown
## [YYYY-MM-DD] Insight Title
- **Context**: What situation led to this insight
- **Lesson**: The key takeaway
- **Applicable to**: When this applies
```
