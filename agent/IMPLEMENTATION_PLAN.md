# Implementation Plan

Normal production sequence:

```bash
cd /home/${USER}/putpocket_dataset_mining
git switch master
git pull --ff-only
./scripts/env/bootstrap_sr.sh --preset server2
source scripts/env/env_activate.sh
putpocket-agent doctor
```

Normal task sequence:

```bash
putpocket-agent task start --topic <topic>
cd <generated-worktree>
source scripts/env/env_activate.sh
<implement/test>
putpocket-agent task close
putpocket-agent task integrate --branch <branch>
```

RunPod equivalents use `/workspace/putpocket_dataset_mining` and `/workspace/putpocket_dataset_mining_worktrees`.
