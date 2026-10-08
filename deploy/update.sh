#!/usr/bin/env bash
# Pulls new code from GitHub, runs ALL tests, and restarts Gimenez only if they pass.
# A failing update is rolled back; the running version keeps trading. (Same approach as TIIM's deploy/update.sh.)
set -uo pipefail
APP=/opt/gimenez
cd "$APP"
as_gz() { sudo -u gimenez "$@"; }

as_gz git fetch --quiet origin || exit 0
BRANCH=$(as_gz git rev-parse --abbrev-ref HEAD)
OLD=$(as_gz git rev-parse HEAD)
NEW=$(as_gz git rev-parse "origin/$BRANCH")
[ "$OLD" = "$NEW" ] && exit 0

echo "Gimenez update: $OLD -> $NEW"
as_gz git merge --ff-only --quiet "origin/$BRANCH" || { echo "update could not be applied cleanly"; exit 1; }
as_gz "$APP/.venv/bin/pip" install -q -r requirements.txt
if as_gz nice -n 15 "$APP/.venv/bin/python" -m pytest -q -x -p no:cacheprovider >/tmp/gimenez-update-tests.log 2>&1; then
  install -m 644 "$APP"/deploy/*.service "$APP"/deploy/*.timer /etc/systemd/system/
  systemctl daemon-reload
  systemctl restart gimenez.service gimenez-research.service gimenez-dashboard.service
  echo "Gimenez updated and restarted: $(as_gz git log -1 --format=%s)"
  sudo -u gimenez "$APP/.venv/bin/python" -m gimenez note "updated to $(as_gz git log -1 --format=%h): $(as_gz git log -1 --format=%s)" || true
else
  echo "Tests FAILED on the update - rolling back, Gimenez keeps running the previous version"
  tail -25 /tmp/gimenez-update-tests.log
  as_gz git reset --hard --quiet "$OLD"
fi
