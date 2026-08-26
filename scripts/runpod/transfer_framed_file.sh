#!/usr/bin/env bash
# Transfer one immutable file through RunPod's interactive SSH proxy in verified chunks.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: transfer_framed_file.sh --ssh-target USER@HOST --identity PATH \
  --remote-file /ABS/PATH --destination PATH --expected-sha256 HEX \
  --expected-bytes N [--chunk-mib N] [--dry-run]

The destination and its .parts directory must not exist. Chunks are retained
after success as transfer evidence. The remote file is read only.
EOF
}

ssh_target=
identity=
remote_file=
destination=
expected_sha256=
expected_bytes=
chunk_mib=256
dry_run=0

while (($#)); do
  case "$1" in
    --ssh-target) ssh_target=${2:?}; shift 2 ;;
    --identity) identity=${2:?}; shift 2 ;;
    --remote-file) remote_file=${2:?}; shift 2 ;;
    --destination) destination=${2:?}; shift 2 ;;
    --expected-sha256) expected_sha256=${2:?}; shift 2 ;;
    --expected-bytes) expected_bytes=${2:?}; shift 2 ;;
    --chunk-mib) chunk_mib=${2:?}; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ $ssh_target =~ ^[A-Za-z0-9._@:-]+$ ]] || { echo "invalid --ssh-target" >&2; exit 2; }
[[ $remote_file =~ ^/[A-Za-z0-9._/+:-]+$ ]] || { echo "invalid --remote-file" >&2; exit 2; }
[[ $expected_sha256 =~ ^[0-9a-f]{64}$ ]] || { echo "invalid --expected-sha256" >&2; exit 2; }
[[ $expected_bytes =~ ^[1-9][0-9]*$ ]] || { echo "invalid --expected-bytes" >&2; exit 2; }
[[ $chunk_mib =~ ^[1-9][0-9]*$ ]] || { echo "invalid --chunk-mib" >&2; exit 2; }
test -n "$identity" && test -n "$destination"

chunk_bytes=$((chunk_mib * 1048576))
chunk_count=$(((expected_bytes + chunk_bytes - 1) / chunk_bytes))
parts_root="${destination}.parts"
candidate="${destination}.candidate"

if ((dry_run)); then
  printf '{"status":"dry_run","remote_file":"%s","expected_bytes":%s,"expected_sha256":"%s","chunk_mib":%s,"chunk_count":%s}\n' \
    "$remote_file" "$expected_bytes" "$expected_sha256" "$chunk_mib" "$chunk_count"
  exit 0
fi

test -f "$identity"
test ! -e "$destination"
test ! -e "$candidate"
test ! -e "$parts_root"
mkdir -p "$(dirname "$destination")" "$parts_root"

transfer_chunk() {
  local index=$1
  local skip_mib=$((index * chunk_mib))
  local expected_part_bytes=$chunk_bytes
  local remaining=$((expected_bytes - index * chunk_bytes))
  local part partial stderr_path observed_bytes
  if ((remaining < expected_part_bytes)); then
    expected_part_bytes=$remaining
  fi
  part=$(printf '%s/part-%05d.bin' "$parts_root" "$index")
  partial="${part}.partial"
  stderr_path=$(printf '%s/part-%05d.ssh.stderr' "$parts_root" "$index")
  test ! -e "$part" && test ! -e "$partial"

  printf '%s\n' \
    'stty -onlcr -echo' \
    "bind 'set enable-bracketed-paste off'" \
    'unset PROMPT_COMMAND' \
    "PS1=''" \
    'set -euo pipefail' \
    "test \"\$(stat -c %s '$remote_file')\" = '$expected_bytes'" \
    "test \"\$(sha256sum '$remote_file' | awk '{print \$1}')\" = '$expected_sha256'" \
    "printf '__PP_%s__\\n' BEGIN" \
    "dd if='$remote_file' bs=1048576 skip='$skip_mib' count='$chunk_mib' iflag=fullblock status=none | base64 -w76" \
    "printf '\\n'" \
    "printf '__PP_%s__\\n' END" \
    'exit' \
  | ssh -tt -i "$identity" "$ssh_target" 2>"$stderr_path" \
  | sed -n '/__PP_BEGIN__/,/__PP_END__/p' \
  | sed '1d;$d' \
  | tr -d '\r' \
  | base64 -d >"$partial"

  observed_bytes=$(stat -c %s "$partial")
  test "$observed_bytes" = "$expected_part_bytes"
  mv "$partial" "$part"
  sha256sum "$part" >"${part}.sha256"
}

for ((index = 0; index < chunk_count; index++)); do
  transfer_chunk "$index"
done

parts=("$parts_root"/part-*.bin)
test "${#parts[@]}" = "$chunk_count"
cat "${parts[@]}" >"$candidate"
test "$(stat -c %s "$candidate")" = "$expected_bytes"
test "$(sha256sum "$candidate" | awk '{print $1}')" = "$expected_sha256"
mv "$candidate" "$destination"
sha256sum "$destination" >"${destination}.sha256"
printf '{"status":"passed","destination":"%s","bytes":%s,"sha256":"%s","chunks_retained":%s}\n' \
  "$destination" "$expected_bytes" "$expected_sha256" "$chunk_count"
