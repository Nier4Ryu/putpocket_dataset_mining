# Single-sample Episode Spec

A single sample attempt is the atomic mining unit.

## Flow

1. Load one MBPP row.
2. Normalize it into a source task.
3. Create an agent-visible Docker workspace containing only `solution.py` stub.
4. Prepare system prompt 1 with Cline rules v1 and query1.
5. Render prompt explicitly with tokenizer chat template.
6. Run history-1 as a multi-step headless Cline trajectory.
7. Snapshot files after history-1.
8. Generate hidden verifier tests from MBPP `test_list`.
9. Run verifier in a fresh container from the snapshot.
10. Prepare system prompt 2 with Cline rules v2, query1, history1, and query2.
11. Render prompt explicitly.
12. Run history-2 as a multi-step headless Cline trajectory.
13. Snapshot files after history-2.
14. Run verifier again in a fresh container from the snapshot.
15. If verification passed, run Codex CLI judge.
16. Save final status and artifacts.

## Status labels

- `accepted`
- `rejected`
- `failed_infra`
- `uncertain`

A sample is complete when one final status and required artifacts are saved.
Accepted means history1 unit tests passed, history2 unit tests passed, and judge returned pass.
