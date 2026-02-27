# Architecture Diagrams

All diagrams are written in [Mermaid](https://mermaid.js.org/) format (`.mmd` files).

## Viewing

**Option 1: GitHub** — GitHub renders `.mmd` files natively in markdown. Just reference them:
```markdown
```mermaid
// paste content here
```‎
```

**Option 2: VS Code** — Install the "Mermaid Markdown Syntax Highlighting" extension.

**Option 3: Online** — Paste into https://mermaid.live

## Diagrams

| File | Description |
|------|-------------|
| `overview.mmd` | **Project Overview** — full system architecture with all layers |
| `retrieval-module.mmd` | **Retrieval Module** — UnifiedRetriever, S2, arXiv internals |
| `agent-module.mmd` | **Agent Module** — ReAct loop, tools, LLM integration |
| `cli-module.mmd` | **CLI Module** — command structure and data flow |

## Legend

- 🟩 Green = Done
- 🟨 Yellow = In Progress
- 🟥 Red = Planned (future phase)
