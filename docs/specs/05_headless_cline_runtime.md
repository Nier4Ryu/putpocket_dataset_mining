# Headless Cline Runtime Spec

The agent is headless Cline.

## Required behavior

- Use original Cline tool format from `externals/cline`.
- Use Cline prompt semantics.
- Use compact Cline prompt by default for `Qwen/Qwen3.5-9B`.
- Do not replace tool calls with custom JSON actions.

## Limits

```yaml
history1_max_turns: 30
history2_max_turns: 30
max_tool_calls_per_turn: follow_original_cline_runtime
max_parse_failures_per_history: 3
```

Parse failure means the model output cannot be parsed as a valid Cline tool call.
Return a format-error observation and continue, up to three failures per history.

## Finish

Use Cline's completion tool semantics, expected as `attempt_completion`.
This ends the rollout but does not determine acceptance; verifier and judge decide acceptance.

## Tools

Tools must operate inside Docker `/workspace`:

- read_file
- write_file / write_to_file
- apply_patch or Cline equivalent
- list_files
- search_file
- execute_command / run_command
- attempt_completion
