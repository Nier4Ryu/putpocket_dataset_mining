# Prompt Rendering and History Spec

## Rule source

Cline rules are not Docker workspace files.
They are static prepared artifacts:

```text
prepared/cline_rules_v1.md
prepared/cline_rules_v2.md
prepared/system_prompt_1.md
prepared/system_prompt_2.md
```

## Message structure

History 1 request:

```text
system_prompt_1, including cline_rules_v1
+ query1
```

History 2 request:

```text
system_prompt_2, including cline_rules_v2
+ query1
+ history1 messages/tool observations
+ query2
```

## Exact rendering

Save all levels:

```text
prepared/messages_history1.json
prepared/messages_history2.json
prepared/rendered_prompt_history1.txt
prepared/rendered_prompt_history2.txt
prepared/tokenization_history1.json
prepared/tokenization_history2.json
```

Putpocket must call tokenizer chat template explicitly before vLLM generation.
vLLM must receive rendered prompt strings, not chat messages requiring internal templating.

## Trajectory

Each history is multi-step:

```text
LLM response
→ Cline tool call
→ Docker workspace observation
→ next LLM response
→ ...
```

Human-readable chronological view must be written to `episode_timeline.md`.
Machine-readable timeline must be written to `episode_timeline.jsonl`.
