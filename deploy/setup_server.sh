#!/usr/bin/env bash
# One-time setup of Gimenez on an Ubuntu server. Run as root:
#   curl -fsSL https://raw.githubusercontent.com/tradingwithedgar-del/autonomous-trader/claude/zealous-heisenberg-xmb8hs/deploy/setup_server.sh | bash
# Safe to run again. It lives NEXT TO TIIM and never touches it: own user (gimenez), own folder
# (/opt/gimenez), own port (8001). Structure adapted from TIIM's deploy/setup_server.sh.
set -euo pipefail

REPO="https://github.com/tradingwithedgar-del/autonomous-trader.git"
BRANCH="${GIMENEZ_BRANCH:-claude/zealous-heisenberg-xmb8hs}"
APP=/opt/gimenez

echo "==> System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl ufw >/dev/null

if ! swapon --show 2>/dev/null | grep -q .; then
  echo "==> No swap found: adding a 2 GB swap file (memory safety buffer)"
  fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
  chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> User 'gimenez' (separate from TIIM)"
id gimenez >/dev/null 2>&1 || useradd --create-home --home-dir /home/gimenez --shell /bin/bash gimenez

echo "==> Downloading Gimenez into $APP"
if [ ! -d "$APP/.git" ]; then
  git clone --quiet -b "$BRANCH" "$REPO" "$APP"
fi
git config --global --add safe.directory "$APP"
mkdir -p "$APP/data"
chown -R gimenez:gimenez "$APP"

echo "==> Python libraries (a few minutes)"
sudo -u gimenez python3 -m venv "$APP/.venv"
sudo -u gimenez "$APP/.venv/bin/pip" install -q --upgrade pip
sudo -u gimenez "$APP/.venv/bin/pip" install -q -r "$APP/requirements.txt"

echo "==> Running Gimenez's tests on this server"
sudo -u gimenez bash -c "cd $APP && nice -n 15 .venv/bin/python -m pytest -q -x -p no:cacheprovider" | tail -3

if [ ! -f "$APP/.env" ]; then
  echo
  echo "==> Your settings (stored only on this server, in $APP/.env, readable only by root and gimenez)"
  read -r -p "TradeLocker DEMO email: " TL_EMAIL </dev/tty
  read -r -s -p "TradeLocker DEMO password: " TL_PASSWORD </dev/tty; echo
  read -r -p "TradeLocker server [PLEXY]: " TL_SERVER </dev/tty; TL_SERVER=${TL_SERVER:-PLEXY}
  read -r -p "Account ID (leave empty if you have only one demo account): " TL_ACC </dev/tty
  read -r -s -p "Choose a password for the Gimenez dashboard: " DASH_PW </dev/tty; echo
  cp "$APP/.env.example" "$APP/.env"
  python3 - "$APP/.env" "$TL_EMAIL" "$TL_PASSWORD" "$TL_SERVER" "$TL_ACC" "$DASH_PW" <<'PY'
import re, sys
path, email, pw, server, acc, dash = sys.argv[1:7]
s = open(path).read()
for key, val in (("TL_EMAIL", email), ("TL_PASSWORD", pw), ("TL_SERVER", server), ("TL_ACC_NUM", acc), ("DASHBOARD_PASSWORD", dash)):
    s = re.sub(rf"^{key}=.*$", lambda m: f"{key}={val}", s, flags=re.M)
open(path, "w").write(s)
PY
  chown gimenez:gimenez "$APP/.env"; chmod 600 "$APP/.env"
fi

echo "==> The 'gimenez' command"
cat > /usr/local/bin/gimenez <<'SH'
#!/usr/bin/env bash
# Run any Gimenez command as the gimenez user, e.g. `gimenez status`, `gimenez why`, `gimenez stop`
cd /opt/gimenez && exec sudo -u gimenez /opt/gimenez/.venv/bin/python -m gimenez "$@"
SH
chmod +x /usr/local/bin/gimenez

echo "==> Checking the TradeLocker connection (places no orders)"
if ! gimenez doctor; then
  echo
  echo "The connection check failed. Fix the settings with:  nano $APP/.env   then run:  gimenez doctor"
  echo "Gimenez is NOT started until the check passes. Re-run this script afterwards."
  exit 1
fi

echo "==> Starting Gimenez 24/7 (trader, research, dashboard, auto-update)"
install -m 644 "$APP"/deploy/*.service "$APP"/deploy/*.timer /etc/systemd/system/
chmod +x "$APP/deploy/update.sh"
systemctl daemon-reload
systemctl enable --now gimenez.service gimenez-research.service gimenez-dashboard.service gimenez-update.timer

ufw allow OpenSSH >/dev/null
ufw allow 8001/tcp >/dev/null
ufw --force enable >/dev/null

IP=$(curl -fsS https://api.ipify.org || hostname -I | awk '{print $1}')
echo
echo "Gimenez is running."
echo "  Dashboard:  http://$IP:8001   (any user name + your dashboard password)"
echo "  Commands:   gimenez status | gimenez why | gimenez strategies | gimenez stop | gimenez resume"
echo "  Logs:       journalctl -u gimenez -f      (research: journalctl -u gimenez-research -f)"
echo
echo "Memory right now (TIIM + Gimenez share this server):"
free -m
