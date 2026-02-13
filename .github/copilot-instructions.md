# Copilot instructions for this repository

This repo is primarily **OpenSpec + Copilot workflow configuration** (custom prompts/skills), not an application codebase.

## Build / test / lint

No build, test, or lint tooling is defined in this repository (no `package.json`, `Makefile`, `pyproject.toml`, etc. detected).

Useful validation commands here are OpenSpec-centric:

```bash
# Check whether OpenSpec is initialized and what it considers “ready/done”
openspec status --json

# List active changes (used by multiple skills/prompts)
openspec list --json
```

## High-level architecture

### 1) Copilot prompts (entry points)
- `/.github/prompts/opsx-*.prompt.md`
  - Markdown prompt files that back the `/opsx:*` slash commands (e.g., `/opsx:new`, `/opsx:apply`, `/opsx:sync`).
  - These prompts describe the intended workflow steps and guardrails at a user-facing level.

### 2) Copilot skills (canonical behavior)
- `/.github/skills/openspec-*/SKILL.md`
  - Skill definitions with YAML front matter (`name`, `description`, etc.) plus detailed step-by-step behaviors.
  - Skills consistently rely on **OpenSpec CLI JSON output** (e.g., `openspec list --json`, `openspec status --json`, `openspec instructions … --json`) and then reading/writing the referenced files.

### 3) OpenSpec project configuration
- `/openspec/config.yaml`
  - Declares the default workflow schema (`spec-driven` currently).
  - Optionally supports project-wide context and per-artifact rules (currently empty placeholders).

### 4) OpenSpec change artifacts (created by the CLI)
When users run OpenSpec commands, the CLI creates and evolves artifacts under:
- `openspec/changes/<change-name>/…`
- Archived changes move under: `openspec/changes/archive/YYYY-MM-DD-<change-name>/…`

For the `spec-driven` schema, the common artifact sequence is:
- `proposal.md` → `specs/<capability>/spec.md` → `design.md` → `tasks.md`

## Key conventions (repo-specific)

### Keep prompts and skills aligned
The repo contains **both** `/.github/prompts/` and `/.github/skills/`. If you change workflow behavior or guardrails, check whether the matching prompt + skill both need updating.

### Prefer OpenSpec CLI JSON outputs as the source of truth
Many workflows are designed to:
- call `openspec … --json`
- parse the result to determine next actions
- read/write the specific `contextFiles` or `outputPath` returned by the CLI

### Change selection should be explicit when ambiguous
Several skills require that when a change name isn’t provided and multiple active changes exist, Copilot should:
- run `openspec list --json`
- use the UI question tool to let the user choose
- avoid guessing/auto-selecting unless the skill explicitly allows it

### Delta-spec syncing behavior
The `/opsx:sync` workflow (see `openspec-sync-specs`) expects delta specs like:
- `openspec/changes/<name>/specs/<capability>/spec.md`
…and applies them into main specs like:
- `openspec/specs/<capability>/spec.md`

It’s an **intelligent merge**, not a wholesale overwrite (e.g., add scenarios without deleting unrelated existing scenarios).

### Repo hygiene while searching
This repo includes a local `.venv/` directory; avoid scanning it unless explicitly needed. Prefer searching within `openspec/` and `.github/` first.
