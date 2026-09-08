#!/usr/bin/env bash
# Put a backup back into a docker-compose studio.
#
#   bash scripts/restore.sh /path/to/backup [/path/to/compose/dir]
#
# Stops the controller, replaces the data volume's contents with the backup,
# starts it again. The backup's own RESTORE.md says the same in words; this
# is the same in commands. Nothing here is clever: it is the thing you want
# to be able to read at three in the morning.
set -euo pipefail
BACKUP="${1:?path to a backup directory}"
COMPOSE_DIR="${2:-$(cd "$(dirname "$0")/../docker" && pwd)}"
[ -f "$BACKUP/MANIFEST.json" ] || { echo "not a backup: no MANIFEST.json in $BACKUP" >&2; exit 1; }
cd "$COMPOSE_DIR"
VOL=$(docker compose config --format json | python3 -c "import sys,json;v=json.load(sys.stdin)['volumes'];print(next(iter(v)))" 2>/dev/null || echo studio-data)
echo "Stopping the controller..."
docker compose stop controller
echo "Restoring into volume $VOL from $BACKUP..."
docker run --rm -v "${COMPOSE_DIR##*/}_${VOL}:/data" -v "$BACKUP:/backup:ro" alpine sh -c '
  set -e
  rm -f /data/studio.db /data/studio.db-wal /data/studio.db-shm
  cp /backup/studio.db /data/studio.db
  [ -f /backup/join_token ] && cp /backup/join_token /data/join_token
  for d in datasets assets artifacts; do
    if [ -d /backup/$d ]; then rm -rf /data/$d; cp -r /backup/$d /data/$d; fi
  done
  echo "restored: $(ls /data)"'
echo "Starting the controller..."
docker compose start controller
echo "Done. Check the studio; runs whose models were not in the backup show as removed."
