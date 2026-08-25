# GLM-5.2 stateful mid-trajectory edit sweep (v3)

This chain evaluates a real edit event, not independent runs that start with the
new system prompt.

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
4. The reuse denominator is every eligible frozen-history token in positions
   `[115, history_token_count)`, including A1. Token 114 is never reused. Q2 and
   every later post-edit extension are always recomputed.
5. Ratio 0 and every nonzero ratio use the official single-instance SWE-bench
   Pro evaluator. The full sweep remains dependent on the ratio-0/10 smoke job.

The transplant hook computes target KV and then overwrites selected rows from
the donor snapshot. Therefore the result is an accuracy ablation only; it makes
no latency, skipped-FLOP, or production-safe gating claim.

The base prompt positions preserve the pre-outcome native DSA ranking. A1 did
not exist during that capture, so its positions receive fixed SHA-256 quantiles
and are interleaved with the base ranking before any ratio outcome is evaluated.
