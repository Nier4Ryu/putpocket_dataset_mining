# Dataset Mining Codex Audit

## 1. Git status
## master
 M tasks/objectives/dataset_mining/state.yaml
?? README.md
?? audit_after_codex_20260707_162835.md
?? docker/
?? docs/DATASET_STRUCTURE.md
?? pyproject.toml
?? scripts/
?? src/
?? tests/

## 2. Git diff stat
 tasks/objectives/dataset_mining/state.yaml | 31 +++++++++++++++++++++++++++++-
 1 file changed, 30 insertions(+), 1 deletion(-)

## 3. Changed files
M	tasks/objectives/dataset_mining/state.yaml

## 4. Objective state
objective_id: dataset_mining
version: v0.5
implementation_lock:
  status: ready_for_implementation
  unlock_value: ready_for_implementation
  note: >
    Leave locked while humans review the feed. Codex must not implement until
    this is changed to ready_for_implementation.

current_status: implementation_complete_runtime_validation_blocked
repo_state:
  expected_repo_kind: standalone_empty_dataset_mining_repo
  owns_own_env: true
  depends_on_fixing_repo: false

decisions:
  resolved:
    - standalone_empty_repo_with_own_env
    - package_name_putpocket_dataset_mining
    - env_activate_sh_name
    - qwen_qwen35_9b_generation_model
    - huggingface_mbpp_google_default_with_muennighoff_fallback
    - mbpp_code_is_reference_solution
    - solution_py_stub_initial_workspace
    - hidden_verifier_tests_only
    - no_workspace_clinerules_file
    - explicit_chat_template_rendering_inside_putpocket
    - deterministic_greedy_generation_temperature_zero
    - mining_seed_default_42
    - RANDOM_SEED_default_42_for_later_evaluation
    - original_cline_tool_format
    - compact_cline_prompt_default_for_9b
    - ubuntu_22_04_python_3_13_docker_base
    - host_mounted_workspace_with_host_uid_gid
    - codex_cli_read_only_judge
    - master_only_sqlite_db_writer
    - parallel_master_worker_multi_sample
    - debug_first_parallel_full_server_profiles
    - local_materialized_dataset_view_not_cross_repo_export
    - shared_server_vllm_build_cpu_cap_32
    - runtime_gpu_devices_restricted_to_4_5_6_7
    - full_server_profile_uses_4_workers_not_8
  remaining_human_decisions: []

next_action_after_unlock: implement_dataset_mining_objective_v0_5

implementation_progress:
  started_by: codex
  status: runtime_validation_blocked
  completed_phases:
    - phase_0_repo_substrate
    - phase_1_env_and_externals_helpers
    - phase_2_docker_default_python_image_spec_and_manager
    - phase_3_dataset_source_adapter
    - phase_4_prompt_rendering
    - phase_5_headless_cline_runtime
    - phase_6_serving_connector
    - phase_7_verifier_and_judge
    - phase_8_single_sample_runner
    - phase_9_multi_sample_master_worker
    - phase_10_dataset_structure_docs
  active_phase: runtime_e2e_validation
  blocker:
    kind: environment_prerequisites_missing
    details:
      - host python3.13 is not on PATH, so Putpocket_env cannot be bootstrapped with the specified Python yet
      - Putpocket_env has not been created
      - externals/vllm, externals/lmcache, and externals/cline have not been checked out
      - local Python used for verification lacks datasets, transformers, and vllm modules
      - default Docker image putpocket-default-python:ubuntu22.04-py313-v1 is not built yet
    validation_completed:
      - python3 -m compileall src tests
      - PYTHONPATH=src python3 -m unittest discover -s tests -v
      - PYTHONPATH=src python3 -m putpocket_dataset_mining.cli doctor

## 5. Remaining blockers / TODO markers
tasks/objectives/dataset_mining/START_HERE.md:23:  status: human_review_required
tasks/objectives/dataset_mining/objective.yaml:120:        content: "# TODO: implement the required function.\n"
configs/dataset_mining/mbpp_stateful_single.yaml:33:    solution.py: "# TODO: implement the required function.\n"
docs/specs/04_docker_workspace.md:48:# TODO: implement the required function.
src/putpocket_dataset_mining/single.py:216:            {"solution.py": "# TODO: implement the required function.\n"},

## 6. Expected implementation footprint
OK      pyproject.toml
OK      scripts/env/env_activate.sh
OK      docker/default_python/Dockerfile
OK      configs/dataset_mining/mbpp_stateful_single.yaml
OK      configs/dataset_mining/mbpp_stateful_multi.yaml
OK      src/putpocket_dataset_mining
OK      tests
OK      docs/DATASET_STRUCTURE.md

## 7. Source tree
src/putpocket_dataset_mining/cline_tools.py
src/putpocket_dataset_mining/cli.py
src/putpocket_dataset_mining/config.py
src/putpocket_dataset_mining/constants.py
src/putpocket_dataset_mining/dataset.py
src/putpocket_dataset_mining/docker_workspace.py
src/putpocket_dataset_mining/doctor.py
src/putpocket_dataset_mining/errors.py
src/putpocket_dataset_mining/externals.py
src/putpocket_dataset_mining/fs.py
src/putpocket_dataset_mining/__init__.py
src/putpocket_dataset_mining/jsonl.py
src/putpocket_dataset_mining/judge.py
src/putpocket_dataset_mining/multi.py
src/putpocket_dataset_mining/prompts.py
src/putpocket_dataset_mining/runtime.py
src/putpocket_dataset_mining/serving.py
src/putpocket_dataset_mining/single.py
src/putpocket_dataset_mining/storage.py
src/putpocket_dataset_mining/verifier.py

## 8. Test tree
tests/test_cline_tools.py
tests/test_dataset_materializer.py
tests/test_multi_preflight.py

## 9. CLI / entrypoint hints
src/putpocket_dataset_mining/cli.py:3:import argparse
src/putpocket_dataset_mining/cli.py:10:def build_parser() -> argparse.ArgumentParser:
src/putpocket_dataset_mining/cli.py:11:    parser = argparse.ArgumentParser(prog="putpocket-dataset-mining")
src/putpocket_dataset_mining/cli.py:122:if __name__ == "__main__":

## 10. Makefile mining targets
missing Makefile

## 11. Data mining artifacts, if any

## 12. Codex run logs, if any
