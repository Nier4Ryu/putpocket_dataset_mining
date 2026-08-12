# Canonical Runtime Policy

The canonical runtime checkout is:

- local: `/home/${USER}/putpocket_dataset_mining`
- RunPod: `/workspace/putpocket_dataset_mining`

It must be on `master` and synchronized to `origin/master` by fast-forward only.

The canonical runtime checkout owns:

- `Putpocket_env`
- canonical `externals/vllm`
- canonical `externals/lmcache`
- production inference
- mining/evaluation
- runtime Docker usage
- Cloud deployment source

It is forbidden to use the canonical runtime checkout for direct feature edits, task branches, or experimental editable installs.
