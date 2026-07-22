# Run GLM evaluation durable goal with Codex

From the repo root:

```bash
cd ~/putpocket_dataset_mining

RUN_DIR="codex_runs/glm_eval_goal_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

cat AGENT_GLM_EVAL_GOAL_PROMPT.md | codex \
  --ask-for-approval never \
  exec \
  --cd . \
  --sandbox danger-full-access \
  --json \
  --output-last-message "$RUN_DIR/final_message.md" \
  "Read the piped /goal prompt. Implement the GLM evaluation runner, run smoke and full evaluation if possible, analyze results, and write TO_GPT.md. Do not stop at smoke-only completion." \
  > "$RUN_DIR/stdout.jsonl" \
  2> "$RUN_DIR/stderr.log"
```

Monitor:

```bash
tail -f "$RUN_DIR/stderr.log" "$RUN_DIR/stdout.jsonl"
```

After completion:

```bash
sed -n '1,240p' TO_GPT.md
find data/model_evaluation/runs -maxdepth 3 -type f | sort | tail -200
```
