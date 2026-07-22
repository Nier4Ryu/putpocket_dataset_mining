from __future__ import annotations

import sys

from putpocket_dataset_mining.constants import GLM52_VLLM025_SERVING_STACK
from putpocket_dataset_mining.model_evaluation import glm_eval


DEFAULT_EVAL_NAME = "eval_glm52_08b_vllm025_on_mbpp_stateful_working_v0"


def _has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _with_v025_defaults(argv: list[str]) -> list[str]:
    patched = list(argv)
    if _has_option(patched, "--limit"):
        patched = ["--max-samples" if arg == "--limit" else arg for arg in patched]
    if not _has_option(patched, "--serving-stack"):
        patched = ["--serving-stack", GLM52_VLLM025_SERVING_STACK, *patched]
    if not _has_option(patched, "--eval-name"):
        patched = ["--eval-name", DEFAULT_EVAL_NAME, *patched]
    return patched


def main(argv: list[str] | None = None) -> int:
    return glm_eval.main(_with_v025_defaults(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
