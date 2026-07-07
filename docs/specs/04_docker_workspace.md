# Docker Workspace Spec

Docker is the simulated user workspace, not merely a unit-test runner.

## Default image

- Base: `ubuntu:22.04`
- Python: 3.13 installed into image
- Image: `putpocket-default-python:ubuntu22.04-py313-v1`
- Dockerfile: `docker/default_python/Dockerfile`

Required packages:

```text
python 3.13
pytest
git
bash
coreutils
ripgrep
tree
jq
```

## Runtime policy

- Runtime network: none
- Runtime dependency install: false
- GPU in workspace container: false
- CPU per episode: 8
- Memory per episode: 8g
- Workspace root: `/workspace`
- Workspace is host-mounted.
- Container runs as host UID/GID.
- Use `HOME=/tmp/putpocket-home` or equivalent writable home.

## Initial workspace

Agent-visible initial files:

```text
/workspace/solution.py
```

with content:

```python
# TODO: implement the required function.
```

No `.clinerules` file.
No visible unit-test file.

## Snapshots

Save file snapshots only, not Docker image snapshots:

```text
initial
after_history1
after_history2
failure
```
