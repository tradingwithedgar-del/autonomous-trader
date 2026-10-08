# Setting up Gimenez (one step at a time)

Gimenez runs on your existing DigitalOcean server (142.93.148.138) next to TIIM. It never touches
TIIM: its own user (`gimenez`), folder (`/opt/gimenez`) and dashboard port (8001).

Do one step, check the result, then move on. Every step works from a Mac Terminal; from a phone use
an SSH app (e.g. Termius) or DigitalOcean's web "Console" button on the droplet page.

## Step 1 - Check the server's memory
```
ssh root@142.93.148.138
free -m
```
Look at the `Mem:` row, `available` column. Gimenez needs about 500 MB.
- Under ~600 MB available: resize the droplet to 2 GB first (DigitalOcean -> droplet -> Resize ->
  CPU and RAM only, so it can be undone). It reboots; TIIM restarts by itself.
- Otherwise continue.

## Step 2 - Install Gimenez
Still logged in to the server:
```
curl -fsSL https://raw.githubusercontent.com/tradingwithedgar-del/autonomous-trader/claude/zealous-heisenberg-xmb8hs/deploy/setup_server.sh | bash
```
It installs, runs all tests, then asks for your PlexyTrade **demo** login (server `PLEXY`) and a new
dashboard password. Then it checks the connection (`gimenez doctor`, no orders placed). Only if that
passes does it start Gimenez. The last thing it prints is the memory use.

## Step 3 - Check the connection result
The doctor output should end with `ALL GOOD` and show your balance, the number of instruments, a gold
quote and some bars. If it fails, fix the settings with `nano /opt/gimenez/.env` (save: Ctrl+O, Enter,
Ctrl+X) and run `gimenez doctor` again.

## Step 4 - Open the dashboard
`http://142.93.148.138:8001` on your phone or Mac. Any user name + your dashboard password.
At first it will say "no edge yet" with no strategies: it is screening markets (about 20-40 minutes),
then downloading history (an hour or two), then researching. Strategies appear only once something
passes every test, and that can take days, or never.

## Step 5 - Daily use
- `gimenez why` - what happened in the last 24 hours and why
- `gimenez status` - the verdict and open trades
- `gimenez stop` - no new real trades (open ones keep their stops); `gimenez resume` to undo
- Logs: `journalctl -u gimenez -f` (Ctrl+C to leave). Research: `journalctl -u gimenez-research -f`
- Updates install themselves every 15 minutes, only if all tests pass on the server.

## If something goes wrong
- Drawdown halt (25%): everything is closed and it stops. Look at the dashboard, then `gimenez resume`.
- Restart: `systemctl restart gimenez`
- Stop completely: `systemctl stop gimenez gimenez-research`
- Remove completely (TIIM untouched): `systemctl disable --now gimenez gimenez-research gimenez-dashboard gimenez-update.timer && rm -rf /opt/gimenez /usr/local/bin/gimenez /etc/systemd/system/gimenez*`
