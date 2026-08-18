#!/usr/bin/env bash
set -euo pipefail

tool=$(basename -- "$0")
case "$tool" in
  nvcc|ptxas) real="/usr/local/cuda/bin/$tool" ;;
  cc|gcc|c++|g++) real="/usr/bin/$tool" ;;
  *) printf 'COMPILER_AUDIT_UNSUPPORTED_TOOL=%s\n' "$tool" >&2; exit 126 ;;
esac
[[ -x $real ]] || { printf 'COMPILER_AUDIT_REAL_TOOL_MISSING=%s\n' "$real" >&2; exit 126; }
log=${PUTPOCKET_COMPILER_AUDIT_LOG:-}
[[ $log == /* ]] || { printf 'COMPILER_AUDIT_LOG_MUST_BE_ABSOLUTE\n' >&2; exit 126; }
python3 - "$log" "$tool" "$real" "$@" <<'PY'
import datetime,fcntl,json,os,re,sys
path,tool,real,*args=sys.argv[1:]
secret=re.compile(r'(token|password|credential|secret|api[_-]?key)',re.I)
clean=[]
for value in args:
    if secret.search(value):
        clean.append(value.split('=',1)[0]+'=<redacted>' if '=' in value else '<redacted>')
    else:
        clean.append(value)
record={'schema_version':1,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'tool':tool,'real_executable':real,'argv':[tool,*clean]}
with open(path,'a',encoding='utf-8') as stream:
    fcntl.flock(stream,fcntl.LOCK_EX)
    stream.write(json.dumps(record,separators=(',',':'),sort_keys=True)+'\n')
    stream.flush(); os.fsync(stream.fileno())
PY
exec "$real" "$@"
