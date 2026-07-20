from __future__ import annotations

import hashlib
import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import DEFAULT_MODEL_ID, SHARED_HF_HUB_CACHE_DIR
from .dataset import SourceTask
from .errors import DependencyError, InfraError


Message = dict[str, str]


COMPACT_CLINE_TOOL_INSTRUCTIONS = """You are Cline, a coding agent working in a Docker-backed user workspace.
Use exactly one original Cline XML tool call per assistant message. Do not emit JSON actions.

Available tools:
<read_file><path>relative/path</path></read_file>
<write_to_file><path>relative/path</path><content>full file content</content></write_to_file>
<replace_in_file><path>relative/path</path><diff>unified diff</diff></replace_in_file>
<list_files><path>.</path><recursive>true</recursive></list_files>
<search_files><path>.</path><regex>pattern</regex><file_pattern>*.py</file_pattern></search_files>
<execute_command><command>pytest -q</command></execute_command>
<attempt_completion><result>brief result</result></attempt_completion>

Only files under /workspace are available. The task implementation belongs in solution.py.
Do not ask for hidden tests and do not create .clinerules files."""


FULL_CLINE_TOOL_INSTRUCTIONS = COMPACT_CLINE_TOOL_INSTRUCTIONS + """

Tool-use policy:
- Inspect files before rewriting them when useful.
- Use write_to_file for complete file replacement.
- Use replace_in_file only with a valid unified diff.
- execute_command runs without network access and cannot install dependencies.
- attempt_completion ends the current history rollout; verification happens later."""


CLINE_RULES_V1 = """# Cline Rules v1

- Work only in the existing Docker workspace.
- Implement the requested MBPP function in solution.py.
- Do not create or read hidden tests.
- Do not create .clinerules or other rule mirror files in the workspace.
- Use only the original Cline XML tool-call format."""


POLICY_DELTAS = {
    "type_hints_required_v1": """# Type Hints Required

All public functions implemented in solution.py must include Python type annotations for parameters and return values.""",
    "google_docstring_required_v1": """# Google Docstring Required

The primary function must include a concise Google-style docstring with Args and Returns sections.""",
    "forbidden_api_v1": """# Forbidden API

Do not use eval, exec, compile, globals, locals, or __import__ in solution.py.""",
}


@dataclass(frozen=True)
class RenderedPrompt:
    rendered_prompt: str
    metadata: dict[str, Any]


class ChatTemplateRenderer:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, cache_dir: Path = SHARED_HF_HUB_CACHE_DIR) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._tokenizer: Any | None = None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise DependencyError("transformers is required for explicit chat-template rendering.") from exc
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                )
            except Exception as exc:  # noqa: BLE001 - include tokenizer load detail in artifacts.
                raise InfraError(f"Failed to load tokenizer for {self.model_id}: {exc}") from exc
        return self._tokenizer

    def render(self, messages: list[Message]) -> RenderedPrompt:
        tokenizer = self.tokenizer
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        prompt_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        metadata = {
            "model_id": self.model_id,
            "chat_template_applied_inside_putpocket": True,
            "vllm_internal_chat_template_allowed": False,
            "tokenizer_class": tokenizer.__class__.__name__,
            "prompt_sha256": prompt_hash,
            "token_count": len(token_ids),
            "add_generation_prompt": True,
        }
        return RenderedPrompt(rendered_prompt=rendered, metadata=metadata)


class QueryBuilder:
    def __init__(self, policy_deltas: list[str] | None = None, mining_seed: int = 42) -> None:
        self.policy_deltas = policy_deltas or [
            "type_hints_required_v1",
            "google_docstring_required_v1",
            "forbidden_api_v1",
        ]
        self.mining_seed = mining_seed

    def build_query1(self, task: SourceTask) -> str:
        api_hint = _public_api_hint(task.reference_solution)
        api_section = f"\nRequired public API:\n{api_hint}\n" if api_hint else ""
        return (
            "Implement the MBPP programming task in solution.py.\n\n"
            "Problem statement:\n"
            f"{task.prompt.strip()}\n\n"
            f"{api_section}"
            "Keep the implementation self-contained and do not add test files."
        )

    def choose_policy_delta(self, task: SourceTask) -> str:
        seed_text = f"{self.mining_seed}:{task.dataset_id}:{task.split}:{task.row_index}:{task.task_id}"
        rng = random.Random(seed_text)
        return rng.choice(self.policy_deltas)

    def build_rules_v2(self, task: SourceTask) -> tuple[str, str]:
        delta_key = self.choose_policy_delta(task)
        delta = POLICY_DELTAS[delta_key]
        rules = CLINE_RULES_V1 + "\n\n" + delta
        return delta_key, rules

    def build_query2(self, task: SourceTask, delta_key: str) -> str:
        if delta_key == "type_hints_required_v1":
            return "Refactor solution.py so the implemented public function has complete type hints while preserving behavior."
        if delta_key == "google_docstring_required_v1":
            return "Refactor solution.py so the primary function has a concise Google-style docstring while preserving behavior."
        if delta_key == "forbidden_api_v1":
            return "Review solution.py and remove any forbidden dynamic execution APIs while preserving behavior."
        return "Refactor solution.py to satisfy the updated rules while preserving behavior."


def _signature_from_node(node: ast.FunctionDef) -> str:
    args = [arg.arg for arg in node.args.posonlyargs + node.args.args]
    if node.args.vararg is not None:
        args.append(f"*{node.args.vararg.arg}")
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg is not None:
        args.append(f"**{node.args.kwarg.arg}")
    return f"{node.name}({', '.join(args)})"


def _public_api_hint(reference_solution: str) -> str:
    try:
        tree = ast.parse(reference_solution)
    except SyntaxError:
        return ""

    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = [
                _signature_from_node(child)
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name == "__init__"
            ]
            suffix = f" with {', '.join(methods)}" if methods else ""
            lines.append(f"- class {node.name}{suffix}")
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            lines.append(f"- function {_signature_from_node(node)}")
    return "\n".join(lines)


class PromptPreparer:
    def __init__(
        self,
        attempt_dir: Path,
        model_id: str,
        profile: str = "compact",
        mining_seed: int = 42,
        renderer: ChatTemplateRenderer | None = None,
    ) -> None:
        self.attempt_dir = attempt_dir
        self.prepared_dir = attempt_dir / "prepared"
        self.model_id = model_id
        self.profile = profile
        self.query_builder = QueryBuilder(mining_seed=mining_seed)
        self.renderer = renderer or ChatTemplateRenderer(model_id=model_id)

    def _tool_instructions(self) -> str:
        return FULL_CLINE_TOOL_INSTRUCTIONS if self.profile == "full" else COMPACT_CLINE_TOOL_INSTRUCTIONS

    def prepare_history1(self, task: SourceTask) -> tuple[list[Message], str]:
        self.prepared_dir.mkdir(parents=True, exist_ok=True)
        query1 = self.query_builder.build_query1(task)
        system_prompt = self._tool_instructions() + "\n\n" + CLINE_RULES_V1
        (self.prepared_dir / "cline_rules_v1.md").write_text(CLINE_RULES_V1, encoding="utf-8")
        (self.prepared_dir / "system_prompt_1.md").write_text(system_prompt, encoding="utf-8")
        (self.prepared_dir / "query1.txt").write_text(query1, encoding="utf-8")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query1},
        ]
        self.save_rendered("history1", messages)
        return messages, query1

    def prepare_history2(
        self,
        task: SourceTask,
        query1: str,
        history1_messages: list[Message],
    ) -> tuple[list[Message], str, str]:
        delta_key, rules_v2 = self.query_builder.build_rules_v2(task)
        query2 = self.query_builder.build_query2(task, delta_key)
        system_prompt = self._tool_instructions() + "\n\n" + rules_v2
        (self.prepared_dir / "cline_rules_v2.md").write_text(rules_v2, encoding="utf-8")
        (self.prepared_dir / "system_prompt_2.md").write_text(system_prompt, encoding="utf-8")
        (self.prepared_dir / "query2.txt").write_text(query2, encoding="utf-8")
        (self.prepared_dir / "query2_metadata.json").write_text(
            json.dumps({"policy_delta": delta_key, "generated_by_mining_loop": True}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        messages: list[Message] = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": query1})
        for message in history1_messages:
            if message.get("role") != "system":
                messages.append(message)
        messages.append({"role": "user", "content": query2})
        self.save_rendered("history2", messages)
        return messages, query2, delta_key

    def save_messages(self, history_name: str, messages: list[Message]) -> None:
        path = self.prepared_dir / f"messages_{history_name}.json"
        path.write_text(json.dumps(messages, indent=2, sort_keys=True), encoding="utf-8")

    def save_rendered(self, history_name: str, messages: list[Message]) -> RenderedPrompt:
        self.save_messages(history_name, messages)
        rendered = self.renderer.render(messages)
        (self.prepared_dir / f"rendered_prompt_{history_name}.txt").write_text(
            rendered.rendered_prompt,
            encoding="utf-8",
        )
        (self.prepared_dir / f"tokenization_{history_name}.json").write_text(
            json.dumps(rendered.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return rendered
