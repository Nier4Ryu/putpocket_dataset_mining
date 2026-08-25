# GLM-5.2 stateful mid-trajectory edit sweep (v3)

This chain evaluates a real edit event, not independent runs that start with the
new system prompt. For authoring new model-neutral SWE-bench or SWE-bench Pro
episodes, use the normative
[`STATEFUL_MID_TRAJECTORY_EDIT_SCENARIO_CONTRACT.md`](STATEFUL_MID_TRAJECTORY_EDIT_SCENARIO_CONTRACT.md).
The GLM v3 experiment below predates that contract and is retained as a legacy
accuracy-emulation reference.

1. The capture job freezes the original 2,071-token equal-length edit contract
   and the base DSA selector evidence.
2. The episode job runs `SYS_old + Q1 -> A1` exactly once, executes the one A1
   shell action, and freezes A1 plus the resulting Q2 observation. It serializes
   `[SYS_old,Q1,A1]` and `[SYS_new,Q1,A1]` and requires that token 114 is their
   only difference.
3. Every ratio starts a clean, identical SWE-bench Pro container. The proxy
   returns the frozen A1 response for turn 1, verifies that executing it produces
   the frozen Q2, then replaces only the system edit sentence before forwarding
   turn 2 to GLM-5.2.
4. The legacy reuse denominator is every eligible frozen-history token in
   positions `[115, history_token_count)`, including A1. Token 114 is mandatory
   target recompute and is never a donor row. Q2 is the frozen pre-edit tool
   observation included in the first post-edit request, but it has no donor KV;
   Q2 and every genuinely later post-edit token run normally after the cache
   patch.
5. Ratio 0 and every nonzero ratio use the official single-instance SWE-bench
   Pro evaluator. The full sweep remains dependent on the ratio-0/10 smoke job.

The transplant hook computes full target KV and then overwrites its legacy
donor-selected rows from the old snapshot. Therefore the result is an
accuracy-emulation ablation only; it is not true selective prefill and makes no
latency, skipped-FLOP, compute-saving, or production-safe gating claim.

The base prompt positions preserve a pre-outcome native DSA ranking oriented to
stability/reuse suitability. A1 did not exist during that capture, so its
positions receive fixed SHA-256 quantiles and are interleaved with that legacy
donor ranking before any ratio outcome is evaluated. These selected positions
are donor rows for this experiment; they MUST NOT be described as the most
important positions to recompute.

New scenarios use the opposite, explicit scientific orientation: rank eligible
positions by `importance_for_recompute`, recompute the top-ranked prefix under
`SYS_new`, and retain old KV only at the unselected complement. If this final
state is emulated with full target prefill plus donor overwrite, the donor rows
are that complement, and the run remains a no-speedup reference backend.
