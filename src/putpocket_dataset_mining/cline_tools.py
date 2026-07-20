from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any

from .docker_workspace import DockerWorkspace
from .errors import ToolParseError


TOOL_NAMES = (
    "read_file",
    "write_to_file",
    "write_file",
    "replace_in_file",
    "apply_patch",
    "list_files",
    "search_files",
    "search_file",
    "execute_command",
    "run_command",
    "attempt_completion",
)


@dataclass(frozen=True)
class ClineToolCall:
    name: str
    params: dict[str, str]
    raw: str


@dataclass(frozen=True)
class ClineToolObservation:
    tool_name: str
    ok: bool
    content: str
    data: dict[str, Any]

    def as_message_content(self) -> str:
        status = "ok" if self.ok else "error"
        return f"<tool_result tool={self.tool_name} status={status}>\n{self.content}\n</tool_result>"


def _extract_params(body: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for match in re.finditer(r"<([a-zA-Z_][\w-]*)>(.*?)</\1>", body, flags=re.DOTALL):
        params[match.group(1)] = html.unescape(match.group(2).strip())
    return params


def parse_cline_tool_calls(text: str) -> list[ClineToolCall]:
    calls: list[ClineToolCall] = []
    for name in TOOL_NAMES:
        pattern = re.compile(rf"<{name}>(.*?)</{name}>", flags=re.DOTALL)
        for match in pattern.finditer(text):
            raw = match.group(0)
            calls.append(ClineToolCall(name=name, params=_extract_params(match.group(1)), raw=raw))
    calls.sort(key=lambda call: text.find(call.raw))
    if not calls:
        raise ToolParseError("No recognized original Cline XML tool call found in model response.")
    return calls


def _bool_param(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes"}


def _apply_search_replace_blocks(original: str, diff: str) -> tuple[bool, str]:
    pattern = re.compile(
        r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        flags=re.DOTALL,
    )
    updated = original
    count = 0
    for match in pattern.finditer(diff):
        search = match.group(1)
        replace = match.group(2)
        if search not in updated:
            return False, f"SEARCH block not found:\n{search}"
        updated = updated.replace(search, replace, 1)
        count += 1
    if count == 0:
        return False, "No Cline SEARCH/REPLACE blocks found."
    return True, updated


def execute_cline_tool(workspace: DockerWorkspace, call: ClineToolCall, timeout_sec: int = 120) -> ClineToolObservation:
    name = call.name
    params = call.params
    try:
        if name == "read_file":
            path = params["path"]
            content = workspace.read_file(path)
            return ClineToolObservation(name, True, content, {"path": path})

        if name in {"write_to_file", "write_file"}:
            path = params["path"]
            content = params.get("content", "")
            workspace.write_file(path, content)
            return ClineToolObservation(name, True, f"Wrote {path}", {"path": path, "bytes": len(content.encode("utf-8"))})

        if name == "replace_in_file":
            path = params["path"]
            diff = params.get("diff", "")
            original = workspace.read_file(path)
            ok, updated_or_error = _apply_search_replace_blocks(original, diff)
            if not ok:
                return ClineToolObservation(name, False, updated_or_error, {"path": path})
            workspace.write_file(path, updated_or_error)
            return ClineToolObservation(name, True, f"Updated {path}", {"path": path})

        if name == "apply_patch":
            diff = params.get("diff") or params.get("patch") or ""
            result = workspace.apply_unified_diff(diff, timeout_sec=timeout_sec)
            return ClineToolObservation(
                name,
                result.returncode == 0,
                result.stdout + result.stderr,
                result.as_dict(),
            )

        if name == "list_files":
            path = params.get("path", ".")
            recursive = _bool_param(params.get("recursive"), default=False)
            files = workspace.list_files(path, recursive=recursive)
            return ClineToolObservation(name, True, json.dumps(files, indent=2), {"path": path, "recursive": recursive, "files": files})

        if name in {"search_files", "search_file"}:
            path = params.get("path", ".")
            regex = params.get("regex") or params.get("query") or ""
            file_pattern = params.get("file_pattern")
            result = workspace.search_files(path, regex, file_pattern=file_pattern, timeout_sec=timeout_sec)
            ok = result.returncode in {0, 1}
            return ClineToolObservation(name, ok, result.stdout + result.stderr, result.as_dict())

        if name in {"execute_command", "run_command"}:
            command = params["command"]
            result = workspace.exec(command, timeout_sec=timeout_sec)
            ok = result.returncode == 0
            return ClineToolObservation(name, ok, result.stdout + result.stderr, result.as_dict())

        if name == "attempt_completion":
            result = params.get("result", "")
            return ClineToolObservation(name, True, result or "attempt_completion received", {"completed": True})

    except Exception as exc:  # noqa: BLE001 - surface tool failure as model observation.
        return ClineToolObservation(name, False, str(exc), {"exception": exc.__class__.__name__})

    return ClineToolObservation(name, False, f"Unsupported tool: {name}", {})
