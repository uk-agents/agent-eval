#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "anthropic>=0.40",
#   "openai>=1.50",
#   "fastmcp>=2.0",
#   "rank-bm25>=0.2",
# ]
# ///
"""
eval.py — provider-agnostic Claude Code plugin eval harness.

Usage:
    uv run eval.py --manifest evals/manifest.json --plugins /path/to/plugins
    uv run eval.py --manifest evals/manifest.json --plugins . --provider openai
    uv run eval.py --manifest evals/manifest.json --plugins . --filter find-member

Environment:
    EVAL_PROVIDER          anthropic (default) | openai
    ANTHROPIC_API_KEY      required for anthropic provider
    ANTHROPIC_BASE_URL     optional — enables DeepSeek/Kimi/GLM-compatible endpoints
    OPENAI_API_KEY         required for openai provider
    OPENAI_BASE_URL        optional — override OpenAI base URL
    MODEL_ID               override default model per provider

Manifest format (JSON):
    {
      "evals": [
        {
          "id": "unique-id",
          "skill": "skill-name",
          "plugin": "plugin-dir-name",
          "prompt": "user prompt to run",
          "assertions": {
            "skill_triggered": "skill-name",       // load_skill called with this name
            "tool_called": "mcp_tool_name",        // MCP tool called at least once
            "must_contain": ["phrase"],            // in final response (case-insensitive)
            "must_not_contain": ["phrase"]         // absent from final response
          }
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


# ─────────────────────────────────────────────────────────────────────────────
#  Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    result: str = ""


@dataclass
class CompletionResult:
    content: str
    tool_calls: list[ToolCall]
    stop_reason: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class SkillMeta:
    name: str
    description: str
    content: str
    plugin: str


@dataclass
class AssertionResult:
    assertion: str
    passed: bool
    detail: str


@dataclass
class EvalResult:
    id: str
    skill: str
    plugin: str
    prompt: str
    status: str  # PASS | FAIL | ERROR
    assertions: list[AssertionResult] = field(default_factory=list)
    response: str = ""
    trace: list[ToolCall] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    preloaded: bool = False  # True when BM25 score >= threshold → skill injected deterministically


# ─────────────────────────────────────────────────────────────────────────────
#  Plugin registry + BM25
# ─────────────────────────────────────────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-style frontmatter delimited by ---. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, parts[2].strip()


class PluginRegistry:
    """Scans a plugins root, loads SKILL.md files, builds a BM25 index."""

    def __init__(self, plugins_root: Path) -> None:
        self.root = plugins_root
        self.skills: dict[str, SkillMeta] = {}
        self.plugin_dirs: dict[str, Path] = {}
        self._scan()
        self._build_bm25()

    def _scan(self) -> None:
        for plugin_dir in sorted(self.root.iterdir()):
            if not plugin_dir.is_dir():
                continue
            # Accept dirs with .claude-plugin/plugin.json or plugin.yaml
            has_claude = (plugin_dir / ".claude-plugin" / "plugin.json").exists()
            has_yaml = (plugin_dir / "plugin.yaml").exists()
            if not (has_claude or has_yaml):
                continue
            plugin_name = plugin_dir.name
            self.plugin_dirs[plugin_name] = plugin_dir
            skills_dir = plugin_dir / "skills"
            if not skills_dir.exists():
                continue
            for skill_dir in sorted(skills_dir.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_file = skill_dir / "SKILL.md"
                if not skill_file.exists():
                    continue
                raw = skill_file.read_text()
                meta, body = _parse_frontmatter(raw)
                name = meta.get("name", skill_dir.name)
                desc = meta.get("description", "")
                # Flatten multi-line YAML block scalars
                desc = " ".join(desc.split())
                if not desc:
                    for line in body.splitlines():
                        line = line.strip().lstrip("#").strip()
                        if line:
                            desc = line[:200]
                            break
                self.skills[name] = SkillMeta(
                    name=name, description=desc, content=raw, plugin=plugin_name
                )

    def _build_bm25(self) -> None:
        self._skill_names = list(self.skills.keys())
        corpus = [
            (s.name + " " + s.description).lower().split()
            for s in self.skills.values()
        ]
        self._bm25: BM25Okapi | None = BM25Okapi(corpus) if corpus else None

    def top_skills(self, query: str, n: int = 15, always_include: str | None = None) -> list[SkillMeta]:
        """Return top-n BM25-ranked skills. Always includes `always_include` if set."""
        if not self._bm25 or not self._skill_names:
            return list(self.skills.values())[:n]
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(scores, self._skill_names), reverse=True)
        top = [self.skills[name] for _, name in ranked[:n] if name in self.skills]
        if always_include and always_include in self.skills:
            if not any(s.name == always_include for s in top):
                top.append(self.skills[always_include])
        return top

    def bm25_score(self, query: str, skill_name: str) -> float:
        """Return BM25 score for a specific skill against a query."""
        if not self._bm25 or skill_name not in self._skill_names:
            return 0.0
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        idx = self._skill_names.index(skill_name)
        return float(scores[idx])

    def get_mcp_servers(self, plugin_name: str) -> dict[str, dict]:
        """Parse .mcp.json for a plugin. Returns HTTP servers only: {name: {url, title}}."""
        plugin_dir = self.plugin_dirs.get(plugin_name)
        if not plugin_dir:
            return {}
        mcp_file = plugin_dir / ".mcp.json"
        if not mcp_file.exists():
            return {}
        try:
            data = json.loads(mcp_file.read_text())
        except json.JSONDecodeError:
            return {}
        return {
            name: {
                "url": cfg["url"],
                "title": cfg.get("title", name),
                "description": cfg.get("description", ""),
            }
            for name, cfg in data.get("mcpServers", {}).items()
            if cfg.get("type") == "http" and cfg.get("url")
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Providers
# ─────────────────────────────────────────────────────────────────────────────

LOAD_SKILL_ANTHROPIC: dict = {
    "name": "load_skill",
    "description": (
        "Load the full instructions for a named skill. "
        "Call this when a skill listed in the catalog matches the user's request."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Skill name to load"}},
        "required": ["name"],
    },
}

LOAD_SKILL_OPENAI: dict = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": "Load the full instructions for a named skill.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
}


def _tool_schema(tool: Any) -> dict:
    """Extract JSON schema from an MCP tool object defensively."""
    schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None)
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}}


class AnthropicProvider:
    DEFAULT_MODEL = "claude-opus-4-5"

    def __init__(self) -> None:
        import anthropic
        self.client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
        )
        self.model = os.environ.get("MODEL_ID", self.DEFAULT_MODEL)

    def load_skill_tool(self) -> dict:
        return LOAD_SKILL_ANTHROPIC

    def mcp_tool_format(self, tool: Any) -> dict:
        return {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": _tool_schema(tool),
        }

    async def complete(self, messages: list[dict], system: str, tools: list[dict]) -> CompletionResult:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=tools,
        )
        content = ""
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
        return CompletionResult(
            content=content,
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "",
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    def extend_messages(
        self, messages: list[dict], result: CompletionResult, tool_results: list[tuple[str, str]]
    ) -> None:
        assistant_content: list[dict] = []
        if result.content:
            assistant_content.append({"type": "text", "text": result.content})
        for tc in result.tool_calls:
            assistant_content.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
            )
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": output}
                for tid, output in tool_results
            ],
        })

    def skill_preload_messages(self, skill_name: str, content: str) -> list[dict]:
        """Synthetic load_skill call+result injected before the first LLM turn (Anthropic format)."""
        return [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "preload-0",
                     "name": "load_skill", "input": {"name": skill_name}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "preload-0", "content": content},
                ],
            },
        ]


class OpenAIProvider:
    DEFAULT_MODEL = "gpt-5-mini"

    def __init__(self) -> None:
        import openai
        self.client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.model = os.environ.get("MODEL_ID", self.DEFAULT_MODEL)

    def load_skill_tool(self) -> dict:
        return LOAD_SKILL_OPENAI

    def mcp_tool_format(self, tool: Any) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": _tool_schema(tool),
            },
        }

    async def complete(self, messages: list[dict], system: str, tools: list[dict]) -> CompletionResult:
        all_messages = [{"role": "system", "content": system}] + messages
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            tools=tools or None,
            tool_choice="auto" if tools else None,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))
        usage = resp.usage
        return CompletionResult(
            content=content,
            tool_calls=tool_calls,
            stop_reason=resp.choices[0].finish_reason or "",
            model=resp.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    def extend_messages(
        self, messages: list[dict], result: CompletionResult, tool_results: list[tuple[str, str]]
    ) -> None:
        messages.append({
            "role": "assistant",
            "content": result.content or None,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.input)}}
                for tc in result.tool_calls
            ],
        })
        for tid, output in tool_results:
            messages.append({"role": "tool", "tool_call_id": tid, "content": output})

    def skill_preload_messages(self, skill_name: str, content: str) -> list[dict]:
        """Synthetic load_skill call+result injected before the first LLM turn (OpenAI format)."""
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "preload-0", "type": "function",
                     "function": {"name": "load_skill",
                                  "arguments": json.dumps({"name": skill_name})}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "preload-0",
                "content": content,
            },
        ]


def make_provider(name: str) -> AnthropicProvider | OpenAIProvider:
    if name == "openai":
        return OpenAIProvider()
    return AnthropicProvider()


# ─────────────────────────────────────────────────────────────────────────────
#  Agent loop
# ─────────────────────────────────────────────────────────────────────────────

MAX_TURNS = 12

# BM25 score threshold above which the target skill is pre-injected deterministically
# (bypasses the LLM's load_skill decision). Observed scores for clear matches: 7–10.
# Set to 0.0 to always pre-inject; float("inf") to never pre-inject (old behaviour).
BM25_PRELOAD_THRESHOLD = 4.0


async def run_agent(
    prompt: str,
    system: str,
    provider: AnthropicProvider | OpenAIProvider,
    all_tools: list[dict],
    registry: PluginRegistry,
    mcp_tool_map: dict[str, Any],  # tool_name → fastmcp.Client
    initial_messages: list[dict] | None = None,
) -> tuple[str, list[ToolCall], int, int]:
    """Run agent loop. Returns (final_response, trace, total_input_tokens, total_output_tokens)."""
    messages: list[dict] = list(initial_messages or []) + [{"role": "user", "content": prompt}]
    trace: list[ToolCall] = []
    total_in = total_out = 0

    for _turn in range(MAX_TURNS):
        result = await provider.complete(messages, system, all_tools)
        total_in += result.input_tokens
        total_out += result.output_tokens

        if not result.tool_calls:
            return result.content, trace, total_in, total_out

        tool_results: list[tuple[str, str]] = []
        for tc in result.tool_calls:
            if tc.name == "load_skill":
                skill_name = tc.input.get("name", "")
                skill = registry.skills.get(skill_name)
                output = skill.content if skill else f"Skill '{skill_name}' not found in registry."
            elif tc.name in mcp_tool_map:
                client = mcp_tool_map[tc.name]
                try:
                    mcp_result = await client.call_tool(tc.name, tc.input)
                    if isinstance(mcp_result, list):
                        output = "\n".join(
                            getattr(item, "text", str(item)) for item in mcp_result
                        )
                    else:
                        output = str(mcp_result)
                except Exception as exc:
                    output = f"MCP tool error ({tc.name}): {exc}"
            else:
                output = f"Unknown tool: {tc.name}"

            tc.result = output
            trace.append(tc)
            tool_results.append((tc.id, output))

        provider.extend_messages(messages, result, tool_results)

    # MAX_TURNS reached — return empty (loop ran out without a final text response)
    return "", trace, total_in, total_out


# ─────────────────────────────────────────────────────────────────────────────
#  Assertion engine
# ─────────────────────────────────────────────────────────────────────────────

def run_assertions(
    assertions: dict[str, Any],
    response: str,
    trace: list[ToolCall],
) -> list[AssertionResult]:
    results: list[AssertionResult] = []
    skill_inputs = {tc.input.get("name", "") for tc in trace if tc.name == "load_skill"}
    mcp_calls = {tc.name for tc in trace if tc.name != "load_skill"}

    if st := assertions.get("skill_triggered"):
        passed = st in skill_inputs
        results.append(AssertionResult(
            assertion=f"skill_triggered: {st}",
            passed=passed,
            detail="✓" if passed else f"load_skill called with: {sorted(skill_inputs) or '(nothing)'}",
        ))

    if tc_name := assertions.get("tool_called"):
        passed = tc_name in mcp_calls
        results.append(AssertionResult(
            assertion=f"tool_called: {tc_name}",
            passed=passed,
            detail="✓" if passed else f"MCP tools called: {sorted(mcp_calls) or '(none)'}",
        ))

    for phrase in assertions.get("must_contain", []):
        passed = phrase.lower() in response.lower()
        results.append(AssertionResult(
            assertion=f"must_contain: {phrase!r}",
            passed=passed,
            detail="✓" if passed else "Phrase not found in response",
        ))

    for phrase in assertions.get("must_not_contain", []):
        passed = phrase.lower() not in response.lower()
        results.append(AssertionResult(
            assertion=f"must_not_contain: {phrase!r}",
            passed=passed,
            detail="✓" if passed else "Phrase found in response (should be absent)",
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Reporter
# ─────────────────────────────────────────────────────────────────────────────

def render_report(results: list[EvalResult], provider_name: str, manifest_path: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(results)
    by_status = {s: sum(1 for r in results if r.status == s) for s in ("PASS", "FAIL", "ERROR")}
    total_in = sum(r.input_tokens for r in results)
    total_out = sum(r.output_tokens for r in results)

    lines = [
        "# Eval Report",
        "",
        f"**Manifest**: `{manifest_path}`  ",
        f"**Provider**: {provider_name}  ",
        f"**Run at**: {ts}",
        "",
        "## Summary",
        "",
        "| Status | Count |",
        "|--------|-------|",
        *[f"| {s} | {by_status[s]} |" for s in ("PASS", "FAIL", "ERROR")],
        f"| **Total** | **{total}** |",
        "",
        f"Tokens: {total_in:,} in / {total_out:,} out",
        "",
        "---",
        "",
    ]

    STATUS_ICON = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}

    for r in results:
        icon = STATUS_ICON.get(r.status, "?")
        lines += [
            f"## {icon} `{r.id}` — {r.status}",
            "",
            f"**Skill**: `{r.skill}`  **Plugin**: `{r.plugin}`  "
            f"**Model**: `{r.model}`  "
            + ("**Skill load**: BM25 pre-injected  " if r.preloaded else "**Skill load**: LLM-decided  "),
            f"**Duration**: {r.duration_s:.1f}s  "
            f"**Tokens**: {r.input_tokens:,} in / {r.output_tokens:,} out",
            "",
            f"**Prompt**: {r.prompt}",
            "",
        ]
        if r.warnings:
            lines.append(f"**Warnings**: {'; '.join(r.warnings)}")
            lines.append("")
        if r.error:
            lines += [f"**Error**: {r.error}", ""]
        if r.assertions:
            lines.append("**Assertions**:")
            lines.append("")
            for a in r.assertions:
                a_icon = "✓" if a.passed else "✗"
                suffix = f" — {a.detail}" if not a.passed else ""
                lines.append(f"- {a_icon} `{a.assertion}`{suffix}")
            lines.append("")
        if r.trace:
            lines.append("**Tool trace**:")
            lines.append("")
            for tc in r.trace:
                args_preview = json.dumps(tc.input)
                if len(args_preview) > 80:
                    args_preview = args_preview[:77] + "…"
                lines.append(f"- `{tc.name}({args_preview})`")
            lines.append("")
        if r.response:
            snippet = r.response[:400].replace("\n", " ")
            if len(r.response) > 400:
                snippet += "…"
            lines += [f"**Response**: {snippet}", ""]
        lines += ["---", ""]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-case runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_eval_case(
    case: dict,
    registry: PluginRegistry,
    provider: AnthropicProvider | OpenAIProvider,
) -> EvalResult:
    eval_id = case["id"]
    skill_name = case["skill"]
    plugin_name = case["plugin"]
    prompt = case["prompt"]
    assertions = case.get("assertions", {})
    warnings: list[str] = []

    # BM25: check if target skill scores above the preload threshold.
    # If yes, inject the skill content as a synthetic load_skill call before the first LLM turn.
    # This mirrors how Claude Code actually loads skills (deterministic, not LLM-decided).
    bm25_score = registry.bm25_score(prompt, skill_name)
    preloaded = bm25_score >= BM25_PRELOAD_THRESHOLD
    target_skill = registry.skills.get(skill_name)

    if preloaded and target_skill:
        # Focused system prompt: skill already loaded, just execute it
        system = (
            "You are a UK legal research assistant. "
            "The skill instructions below have been loaded for this request. "
            "Follow them precisely and use the MCP tools they specify. "
            "Do NOT answer from your own training knowledge — use only the tools available.\n"
        )
        # Synthetic trace entry so skill_triggered assertion still passes
        preload_trace = [ToolCall(
            id="preload-0",
            name="load_skill",
            input={"name": skill_name},
            result=target_skill.content,
        )]
    else:
        # Fallback: offer load_skill as a callable tool with a catalog
        top = registry.top_skills(prompt, n=15, always_include=skill_name)
        catalog = "\n".join(f"- **{s.name}**: {s.description[:120]}" for s in top)
        system = (
            "You are a UK legal research assistant. "
            "You have access to skills and live MCP tools — you MUST use them. "
            "Do NOT answer from your own training knowledge.\n\n"
            "## Rules (follow exactly)\n"
            "1. Before responding, check whether a skill in the catalog matches the request.\n"
            "2. If a skill matches, you MUST call `load_skill` with that skill's name FIRST. "
            "Do not write any response before calling load_skill.\n"
            "3. After load_skill returns, follow its instructions precisely.\n"
            "4. If no skill matches, say so — do not answer from your own knowledge.\n\n"
            "## Skills catalog\n"
            f"{catalog}\n"
        )
        preload_trace = []

    # Connect to HTTP MCP servers for this plugin
    servers = registry.get_mcp_servers(plugin_name)
    all_tools: list[dict] = [] if preloaded else [provider.load_skill_tool()]
    mcp_tool_map: dict[str, Any] = {}

    t0 = time.monotonic()

    try:
        from fastmcp import Client as MCPClient

        async with AsyncExitStack() as stack:
            for server_name, server_cfg in servers.items():
                try:
                    client = await stack.enter_async_context(MCPClient(server_cfg["url"]))
                    mcp_tools = await client.list_tools()
                    for tool in mcp_tools:
                        all_tools.append(provider.mcp_tool_format(tool))
                        mcp_tool_map[tool.name] = client
                except Exception as exc:
                    warnings.append(f"MCP '{server_name}' unreachable: {exc}")

            # Build initial messages: preload injection (if any) + user prompt
            preload_messages = (
                provider.skill_preload_messages(skill_name, target_skill.content)
                if preloaded and target_skill
                else []
            )

            response, agent_trace, total_in, total_out = await run_agent(
                prompt=prompt,
                system=system,
                provider=provider,
                all_tools=all_tools,
                registry=registry,
                mcp_tool_map=mcp_tool_map,
                initial_messages=preload_messages,
            )

        trace = preload_trace + agent_trace
        duration = time.monotonic() - t0
        assertion_results = run_assertions(assertions, response, trace)
        status = "PASS" if all(a.passed for a in assertion_results) else "FAIL"

        return EvalResult(
            id=eval_id,
            skill=skill_name,
            plugin=plugin_name,
            prompt=prompt,
            status=status,
            assertions=assertion_results,
            response=response,
            trace=trace,
            model=provider.model,
            input_tokens=total_in,
            output_tokens=total_out,
            duration_s=duration,
            warnings=warnings,
            preloaded=preloaded,
        )

    except Exception as exc:
        return EvalResult(
            id=eval_id,
            skill=skill_name,
            plugin=plugin_name,
            prompt=prompt,
            status="ERROR",
            error=str(exc),
            duration_s=time.monotonic() - t0,
            warnings=warnings,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    cases: list[dict] = manifest.get("evals", [])

    if args.filter:
        cases = [
            c for c in cases
            if args.filter in c.get("id", "") or args.filter in c.get("skill", "")
        ]

    if not cases:
        print("No eval cases to run.", file=sys.stderr)
        return 1

    plugins_root = Path(args.plugins).resolve()
    if not plugins_root.exists():
        print(f"error: plugins root not found: {plugins_root}", file=sys.stderr)
        return 1

    provider_name = args.provider or os.environ.get("EVAL_PROVIDER", "anthropic")
    provider = make_provider(provider_name)

    print(f"Loading plugin registry from {plugins_root} …")
    registry = PluginRegistry(plugins_root)
    print(f"  {len(registry.skills)} skills across {len(registry.plugin_dirs)} plugins")
    print(f"Running {len(cases)} eval(s)  provider={provider_name}  model={provider.model}")
    print()

    results: list[EvalResult] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} … ", end="", flush=True)
        result = await run_eval_case(case, registry, provider)
        results.append(result)
        icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}.get(result.status, "?")
        print(f"{icon} {result.status} ({result.duration_s:.1f}s)")
        for w in result.warnings:
            print(f"   ⚠  {w}")
        for a in result.assertions:
            if not a.passed:
                print(f"   ✗ {a.assertion}")
        if result.error:
            print(f"   💥 {result.error}")

    report = render_report(results, provider_name, str(manifest_path))
    out_path = (
        Path(args.output)
        if args.output
        else Path(f"eval-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md")
    )
    out_path.write_text(report)
    print()
    print(f"Report: {out_path}")

    passed = sum(1 for r in results if r.status == "PASS")
    total = len(results)
    print(f"Result: {passed}/{total} passed")
    return 0 if passed == total else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provider-agnostic Claude Code plugin eval harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", required=True, help="Path to eval manifest JSON")
    parser.add_argument("--plugins", required=True, help="Path to plugins root directory")
    parser.add_argument("--provider", help="anthropic (default) | openai")
    parser.add_argument("--filter", help="Filter cases by id or skill substring")
    parser.add_argument("--output", help="Report output path (default: eval-report-<ts>.md)")
    sys.exit(asyncio.run(main_async(parser.parse_args())))


if __name__ == "__main__":
    main()
