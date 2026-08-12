# Integration Policy

Integration is fast-forward only.

`putpocket-agent task integrate` must:

1. fetch origin;
2. verify `origin/master` is an ancestor of the task branch;
3. push the task tip to `refs/heads/master` without force;
4. run `putpocket-agent runtime sync`;
5. verify canonical runtime source ownership.

Never force-push or rewrite implementation history.

Do not update or delete `blackwell` as part of normal task integration.
