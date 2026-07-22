# Cline Prompt Architecture Audit — Analysis Only

You are not implementing a patch in this run.

You are an architecture auditor for the dataset-mining repo.

The current repo appears to work at runtime, but we suspect that the Cline prompt implementation is architecturally wrong. The problem is not simply that a prompt is short. The suspected problem is that the current implementation may have confused or mixed together these separate concepts:

1. Cline base system prompt
2. Cline compact prompt profile
3. Cline full prompt profile
4. Cline tool-call format
5. Cline rules / project rules
6. Dataset-specific runtime contract
7. MBPP-specific workspace contract
8. Query wrapper / dataset agentic wrapper
9. Verifier / hidden test policy
10. Prompt rendering / chat-template rendering artifacts

Your job is to analyze this carefully and produce a human-reviewable report.

Do not modify files.
Do not apply patches.
Do not create new source modules.
Do not rewrite prompts.
Do not edit objective.yaml or state.yaml.
Do not run long dataset mining jobs.
Do not run vLLM, Docker builds, or multi-sample mining.

You may inspect files and run read-only shell commands such as:
- git status
- rg
- sed
- cat
- find
- wc
- python one-liners that only read files

If you believe a code change is obvious, do not make it.
Instead, explain:
- why it is needed,
- what files would be affected,
- what human decisions are required before implementing it,
- what tests would prove it is correct.

## Context

Current implementation has already completed dataset mining validation, but we now suspect prompt architecture issues.

Known suspicious areas:
- src/putpocket_dataset_mining/prompts.py
- src/putpocket_dataset_mining/cline_tools.py
- src/putpocket_dataset_mining/runtime.py
- configs/dataset_mining/mbpp_stateful_single.yaml
- configs/dataset_mining/mbpp_stateful_multi.yaml
- docs/specs/
- tasks/objectives/dataset_mining/objective.yaml
- tasks/objectives/dataset_mining/state.yaml
- externals/cline/

Specific concern:
The issue is not “Cline rules are too short.”
The issue is likely:
“The repo implemented the Cline system prompt incorrectly by mixing Cline prompt profiles with MBPP-specific dataset runtime constraints.”

Examples to investigate:
- FULL_CLINE_TOOL_INSTRUCTIONS may not be a real full Cline prompt.
- COMPACT_CLINE_TOOL_INSTRUCTIONS may include MBPP/Docker-specific strings such as /workspace and solution.py.
- .clinerules may be treated as both prompt artifact and workspace contract inconsistently.
- replace_in_file prompt text and parser behavior may not match original Cline.
- compact/full profile may be implemented as string constants instead of template-based prompt builders.
- dataset runtime contract may not be represented as a separate object/layer.
- MBPP-specific query wrapping may be embedded in generic Cline prompt text.
- hidden verifier test policy may be mixed into Cline base instructions.

## Required investigation

### 1. Current implementation map

Find and summarize:
- where Cline prompt strings are defined,
- where compact/full profile is selected,
- where Cline rules v1/v2 are defined,
- where MBPP-specific strings are introduced,
- where query1/query2 are generated,
- where rendered prompts and tokenization metadata are saved,
- where original Cline tool calls are parsed,
- where replace_in_file is parsed,
- where workspace paths such as /workspace and solution.py are introduced.

For each item, include file paths and short evidence snippets.

### 2. Cline reference audit

Inspect externals/cline and identify the best candidate source files for:
- compact prompt,
- full prompt,
- system prompt registry or builder,
- tool descriptions,
- replace_in_file format,
- attempt_completion / finish behavior,
- relevant local-model prompt behavior.

Do not assume. Search the repo and cite exact file paths.

Questions to answer:
- Does externals/cline contain a compact prompt source?
- Does externals/cline contain a full prompt builder or snapshot?
- Does original Cline mention /workspace or solution.py in a generic system prompt?
- What replace_in_file format does original Cline currently describe?
- Is there a difference between legacy and current replace_in_file formats?

### 3. Root cause analysis

Explain why the current design is wrong or risky.

Do not frame this as “prompt is too short.”
Frame it in terms of:
- wrong abstraction boundary,
- profile/runtime contract confusion,
- dataset-specific text leaking into generic Cline prompt,
- risk to future dataset support such as SWE-bench,
- risk to prompt fidelity claims,
- risk to existing mined dataset interpretation.

### 4. Similar mistake scan

Search for other places where these layers may be mixed:
- Cline prompt vs MBPP runtime contract
- Cline rules vs dataset policy deltas
- query wrapper vs system prompt
- verifier policy vs agent-visible prompt
- Docker workspace assumptions vs generic agent prompt
- hidden tests vs user-visible task instructions
- model-specific prompt rendering vs generic prompt construction

Produce a table:
| Suspected issue | File path | Evidence | Severity | Needs human decision? |

### 5. Correct target architecture

Propose a target architecture, but do not implement it.

The architecture should separate at least these layers:
- ClinePromptBuilder
- ClinePromptProfile: compact/full
- ClineRulesProvider
- DatasetRuntimeContract
- DatasetAgenticWrapper
- PromptComposer / PromptPreparer
- ToolFormatParser
- PromptSourceManifest artifact

For each layer:
- define responsibility,
- define what must not be inside it,
- list likely files/classes to create or modify.

### 6. Human decision list

Before coding, list all human decisions required.

The list must be concrete and answerable.

Examples:
- Should full Cline prompt be implemented now or marked unsupported?
- Should compact be ported from externals/cline or keep current compact_v0?
- Should dataset runtime overlay be placed in system prompt or user query?
- Should replace_in_file support only current Cline format or both current and legacy markers?
- Should existing mbpp_stateful_working_v0 be considered legacy prompt schema?
- Should new mined data use a new dataset version after prompt refactor?
- Should SWE-bench support be interface-only for now?

Do not hide decisions inside implementation assumptions.

### 7. Implementation plan proposal

After the human decision list, propose a staged implementation plan.

The plan must have:
- stages,
- files likely touched,
- tests to add,
- acceptance commands,
- rollback risks,
- whether existing mined dataset remains valid or should be marked legacy.

Do not patch.
Do not edit files.

### 8. Final output format

Return the report in this structure:

# Prompt Architecture Audit Report

## Executive Summary

## Current Implementation Map

## Cline Reference Audit

## Root Cause Analysis

## Similar Mistake Scan

## Target Architecture

## Required Human Decisions

## Proposed Implementation Plan

## Acceptance Criteria For Future Patch

## Risks / Unknowns

## Appendix: Commands Run

If you cannot answer a section, explicitly say what information is missing and how to obtain it.
