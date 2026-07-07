# Verifier and Judge Spec

## Hidden verifier tests

MBPP tests are not visible to the agent.
The verifier materializes MBPP `test_list` into pytest only inside the verifier container.

Flow:

```text
after_history1 snapshot
→ fresh verifier container
→ inject /workspace/tests/test_solution.py
→ pytest -q tests/test_solution.py
→ save checklist/stdout/stderr
```

Repeat after history2.

## Checklist

Save:

```text
verification/history1/checklist.json
verification/history2/checklist.json
```

Each checklist must include stage, checks, final_status, and failure_class.
Failure classes must include history/stage labels.

## Judge

Judge backend is Codex CLI, not local vLLM.

Policy:

```yaml
sandbox: read-only
approval: never
skip_if_unit_test_failed: true
```

Judge input scope:

- cline_rules_v1
- files_after_history1
- cline_rules_v2
- query2
- files_after_history2
- history2_unit_test_summary

Judge rubric:

Return pass only if files after history2 appear to satisfy query2 while following cline-rules v2.

Judge output required schema:

```json
{"decision": "pass|fail|uncertain"}
```
