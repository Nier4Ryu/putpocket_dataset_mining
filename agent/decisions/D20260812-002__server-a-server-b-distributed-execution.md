# D20260812-002: Server-A / Server-B distributed execution

status: accepted
date: 2026-08-12

## Decision

Project-wide, “Server-A inference -> Server-B verification” means:

- Server-A owns model inference, the controller, Query-1/Query-2 generation, and inference/KV/timing research.
- Server-B owns isolated agent-workspace execution and hidden verification.
- Model-generated commands execute only inside a dedicated Server-B workspace container reached through a trusted structured wrapper.
- Hidden tests execute in a separate fresh verifier container. Verification-1 is pytest-only; Verification-2 runs pytest and then Codex Judge only after pytest passes.
- History-1 and History-2 share one logical workspace session and filesystem lineage.
- Server-A local Docker is not a prerequisite unless an execution profile explicitly selects `local_docker`.

The role names are architectural and are not machine names. Deployments map concrete hosts through configuration. The current RunPod/Server-1 profile maps RunPod to Server-A and Server-1 (`cerrotorre`) to Server-B.

## Interfaces and boundaries

The existing workspace interface (`exec`, file operations, search/diff, lifecycle) remains canonical. `ssh_remote_docker` implements it using `putpocket-remote-workspace`, with a durable session ID, a persistent isolated Docker container, and a dedicated remote workspace root. The SSH process may invoke the trusted wrapper; arbitrary model-generated commands are accepted only as structured input and passed to `docker exec`, never to a Server-B host shell.

Workspace and verifier transports may share an SSH endpoint, ProxyJump route, identity, and known-hosts file, but use distinct wrappers and roots. Remote workspace snapshots are immutable verifier inputs; hidden tests are never injected into the mutable agent workspace.

Persisted scheduler state may retain logical endpoint configuration and session identifiers, but must never persist credentials, SSH sockets, transient Pod addresses, or container IDs as identity.

## Consequences

- Preflight dispatches Docker checks according to the selected workspace backend.
- `local_docker` checks Docker on Server-A.
- `ssh_remote_docker` skips Server-A Docker and checks the Server-B workspace wrapper, protocol, daemon, image, root, and disposable lifecycle independently from verifier/Judge preflight.
- Sequential execution is the first live acceptance mode. Pipeline and Staged Forward retain serializable logical workspace identity for future scheduling work; no KV continuity is implied.
