# GLM-5.2 offset-shift fixture decision

## Selected scenario

`mbpp_730_duplicate_user_turn_delete_v1` deletes message index 2, the second
identical Query1 user turn, from
`/home/dyryu/attempt_5a8d1db9b812/prepared/messages_history2.json`.
The source is accepted MBPP train task 730 (`row_index=129`) from
`google-research-datasets/mbpp`, attempt `attempt_5a8d1db9b812`.  The edit is
conversation de-duplication, not a system, identity, authorization, security,
or tool-permission change.  The first copy of the task remains.  Neither model
request contains the source-task reference solution or hidden tests.

The exact tokenizer is `nvidia/GLM-5.2-NVFP4` revision
`aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`; all three tokenizer file digests
are pinned by the fixture builder.  Chat-template and special tokens are
included and `add_generation_prompt=true`.

Exact token geometry:

- old/new length: 909 / 861
- deleted old span: `[349,397)` (48 tokens)
- edited empty landing span: `[349,349)`
- exact aligned suffix: old `[397,909)` to new `[349,861)` (512 tokens)
- suffix absolute-position delta (`new-old`): -48
- old/new prompt-token SHA256:
  `2fbd9d50825827d748ff56844d87941f1beba7b344b3e6e730905044416f16ae` /
  `3dcceb2142dca9bb6272a2d677b32a3bd565ea17b1bdd645515a2c3bcf301297`
- real 64-token page count: 15 / 14
- deletion touches old pages 5 and 6; the new landing point is page 5
- aligned suffix begins on old page 6 and new page 5

## Candidate comparison

The generated `candidate-inventory.json` is authoritative for exact spans,
token IDs, serialized baseline/edited texts, page effects, and hashes.  The
plausible candidates were:

| candidate | tokens old->new | delta | aligned suffix | decision |
| --- | ---: | ---: | ---: | --- |
| delete duplicate Query1 | 909->861 | -48 | 512 | selected; non-control redundancy |
| insert duplicate Query1 (inverse) | 861->909 | +48 | 512 | useful inverse, but deletion reflects the observed source defect |
| delete write tool round | 909->767 | -142 | 366 | rejected; removes task-state transition |
| delete execute tool round | 909->788 | -121 | 246 | rejected; removes verification evidence |
| delete format recovery round | 909->716 | -193 | 53 | rejected; tool-protocol control critical |
| delete completion round | 909->875 | -34 | 22 | rejected; little downstream evidence |
| delete Query2 | 909->889 | -20 | 2 | rejected; deletes active semantic task |
| delete docstring system policy | 909->886 | -23 | 608 | rejected; system/control critical |

## Fail-closed reuse classification

The deletion landing page and one immediate 64-token page neighbour on each
side form local physical/dependency closure: new pages 4, 5, and 6, positions
`[256,448)`.  Pages 0-3, positions `[0,256)`, precede the edit and have zero
position delta, so they are the only raw donor candidates under the present
evidence.

Every new token `[349,861)` maps to an identical donor token 48 positions later,
but its hidden state is causally downstream of the edit and its RoPE coordinate
differs.  With no matching native stability capture and no validated RoPE
correction, full fail-closed closure recomputes `[256,861)`.  The artifact lists
new `[448,861)` / old `[496,909)` separately as a 413-token geometry-only
RoPE-correctable candidate and as a forced raw unsafe-ablation candidate.  It is
not a safe set.

The existing runtime hook copies raw same-position MLA latent and packed
indexer rows only.  It has no old-to-new position gather, no MLA/indexer RoPE
transform, and no shifted block-table contract.  The sweep runner now accepts
the fixture contract explicitly but rejects before HTTP/GPU work with one of:

- `SHIFTED_RAW_KV_REUSE_UNSAFE_AND_UNSUPPORTED`
- `ROPE_CORRECTION_RUNTIME_NOT_IMPLEMENTED`
- `SHIFTED_RUNTIME_POSITION_MAP_NOT_IMPLEMENTED`

The old 2071-token equal-token edit remains the default control.  Production
and runtime mutation defaults remain OFF.

