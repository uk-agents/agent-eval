# agent-eval

Provider-agnostic eval harness for Claude Code plugins. Runs a manifest of prompts through an agent loop, asserts on skill triggering and MCP tool calls, and produces a markdown report.

## Quick start

```bash
cp .env.example .env
# edit .env with your API key

uv run eval.py \
  --manifest showcase/skill-trigger.json \
  --plugins /path/to/uk-legal-plugins
```

No install step — `uv run` handles dependencies via the PEP 723 header in `eval.py`.

## Usage

```
uv run eval.py --manifest <path> --plugins <path> [options]

Options:
  --provider   anthropic (default) | openai
  --filter     Run only cases whose id or skill contains this substring
  --output     Report output path (default: eval-report-<timestamp>.md)
```

## Manifest format

```json
{
  "evals": [
    {
      "id": "unique-id",
      "skill": "skill-name",
      "plugin": "plugin-directory-name",
      "prompt": "user prompt",
      "assertions": {
        "skill_triggered": "skill-name",
        "tool_called": "mcp_tool_name",
        "must_contain": ["[uk-legal MCP"],
        "must_not_contain": ["fabricated phrase"]
      }
    }
  ]
}
```

### Assertion types

| Key | Passes when |
|---|---|
| `skill_triggered` | `load_skill` called with this skill name |
| `tool_called` | Named MCP tool called at least once in the trace |
| `must_contain` | Phrase present in final response (case-insensitive) |
| `must_not_contain` | Phrase absent from final response |

## How it works

1. **PluginRegistry** scans the plugins root, loads all `SKILL.md` files, and builds a BM25 index over names + descriptions.
2. For each eval case, the top-15 BM25 results (always including the target skill) are injected into the system prompt as a skill catalog.
3. The agent can call `load_skill(name)` to get full skill instructions, and any MCP tools from the plugin's `.mcp.json`.
4. HTTP MCP servers are connected via `fastmcp.Client`. Stdio servers are skipped. Unreachable servers produce a warning but the eval still runs.
5. Assertions are checked against the final response and the tool call trace.
6. A markdown report is written with per-case results, traces, and token counts.

## Showcase manifests

| File | What it tests |
|---|---|
| `showcase/skill-trigger.json` | Skill triggering — no MCP required |
| `showcase/anti-fabrication.json` | Agent doesn't invent citations when MCP returns nothing — requires `uk-legal-mcp.fly.dev` |

## Environment

See `.env.example`. Key variables:

| Var | Purpose |
|---|---|
| `EVAL_PROVIDER` | `anthropic` (default) or `openai` |
| `ANTHROPIC_API_KEY` | Required for Anthropic |
| `ANTHROPIC_BASE_URL` | DeepSeek / Kimi / GLM-compatible endpoints |
| `OPENAI_API_KEY` | Required for OpenAI |
| `MODEL_ID` | Override default model (`claude-opus-4-5` / `gpt-4o`) |

## Adding evals to a plugin repo

Create `evals/manifest.json` inside the plugin repository and run:

```bash
uv run /path/to/agent-eval/eval.py \
  --manifest evals/manifest.json \
  --plugins .
```

See `uk-agents/uk-legal-plugins` for the reference eval manifest.
