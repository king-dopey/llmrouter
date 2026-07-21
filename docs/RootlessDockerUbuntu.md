## Ubuntu rootless Docker setup for MCP servers

This procedure installs and configures **rootless Docker** on Ubuntu for a single non-root user that will run MCP servers.

## Result

After completing these steps, the target user can:

- run `docker` without `sudo`
- have Docker start automatically after reboot
- run MCP server containers as a non-root user
- use the rootless Docker socket at `/run/user/$UID/docker.sock`

## Assumptions

- Ubuntu host
- one existing Ubuntu user will own and run Docker, example: `heaps`
- you have `sudo` access for initial setup
- Docker should run rootless, not via the `docker` group

## What was configured

| Area | Final configuration |
|---|---|
| Docker install method | Official Docker apt repository |
| Execution model | Rootless Docker |
| Service manager | `systemctl --user` |
| Auto-start after reboot | `systemctl --user enable docker` + `loginctl enable-linger` |
| Docker socket | `unix:///run/user/$UID/docker.sock` |
| Storage backend | `fuse-overlayfs` |
| Snapshotter setting | `"containerd-snapshotter": false` |
| Log driver | `local` |

## Step 1: install Docker and prerequisites

Run as a sudo-capable user:

```bash
TARGET_USER=heaps

sudo apt update
sudo apt install -y ca-certificates curl uidmap dbus-user-session fuse-overlayfs

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

cat <<EOF | sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin \
  docker-ce-rootless-extras
```

## Step 2: ensure subordinate UID/GID mappings exist

Run as a sudo-capable user:

```bash
TARGET_USER=heaps

SUBUID_START="$(awk -F: 'BEGIN{max=100000} {end=$2+$3; if (end>max) max=end} END{print max}' /etc/subuid 2>/dev/null)"
SUBGID_START="$(awk -F: 'BEGIN{max=100000} {end=$2+$3; if (end>max) max=end} END{print max}' /etc/subgid 2>/dev/null)"

grep -q "^${TARGET_USER}:" /etc/subuid || echo "${TARGET_USER}:${SUBUID_START}:65536" | sudo tee -a /etc/subuid
grep -q "^${TARGET_USER}:" /etc/subgid || echo "${TARGET_USER}:${SUBGID_START}:65536" | sudo tee -a /etc/subgid

grep "^${TARGET_USER}:" /etc/subuid
grep "^${TARGET_USER}:" /etc/subgid
```

Expected format:

```text
heaps:231072:65536
```

## Step 3: disable the system-wide rootful Docker daemon

Run as a sudo-capable user:

```bash
sudo systemctl disable --now docker.service docker.socket || true
sudo rm -f /var/run/docker.sock
```

## Step 4: enable lingering for the target user

This is required so the user’s Docker service can start automatically after reboot even without an interactive login.

Run as a sudo-capable user:

```bash
sudo loginctl enable-linger heaps
```

## Step 5: log in as the target user

Use a normal login or SSH session as that user:

```bash
ssh heaps@your-host
```

Or locally:

```bash
su - heaps
```

## Step 6: install the rootless Docker user service

Run as the target user:

```bash
dockerd-rootless-setuptool.sh install
```

## Step 7: configure the Docker client to use the rootless socket

Run as the target user:

```bash
grep -qxF 'export DOCKER_HOST=unix:///run/user/$UID/docker.sock' ~/.bashrc || \
  echo 'export DOCKER_HOST=unix:///run/user/$UID/docker.sock' >> ~/.bashrc

export DOCKER_HOST=unix:///run/user/$UID/docker.sock
```

## Step 8: configure the rootless daemon storage backend

Run as the target user:

```bash
mkdir -p ~/.config/docker

cat > ~/.config/docker/daemon.json <<'EOF'
{
  "features": {
    "containerd-snapshotter": false
  },
  "storage-driver": "fuse-overlayfs",
  "log-driver": "local"
}
EOF
```

## Step 9: enable and start Docker so it survives reboot

This step must use `enable --now`, not just `start`.

Run as the target user:

```bash
systemctl --user daemon-reload
systemctl --user enable --now docker
```

## Step 10: verify the installation

Run as the target user:

```bash
docker info
docker run --rm hello-world
```

You should see:

- `rootless` in Docker security options
- `Storage Driver: fuse-overlayfs`
- successful `hello-world` output

## Verify automatic startup after reboot

Run these checks:

```bash
systemctl --user is-enabled docker
loginctl show-user "$(whoami)" -p Linger
echo "$DOCKER_HOST"
```

Expected output:

- `enabled`
- `Linger=yes`
- `unix:///run/user/<uid>/docker.sock`

## Expected file locations

| Item | Location |
|---|---|
| User Docker service | `~/.config/systemd/user/docker.service` |
| Rootless daemon config | `~/.config/docker/daemon.json` |
| Rootless Docker data | `~/.local/share/docker` |
| Rootless socket | `/run/user/$UID/docker.sock` |

## MCP server container examples

### HTTP/SSE MCP server

Use high ports such as `8080` and bind to localhost unless remote access is required.

```bash
mkdir -p "$HOME/mcp-data" "$HOME/mcp-config"

docker run -d \
  --name mcp-server \
  --restart unless-stopped \
  -p 127.0.0.1:8080:8080 \
  -v "$HOME/mcp-data:/data" \
  -v "$HOME/mcp-config:/config:ro" \
  IMAGE:TAG
```

### stdio-based MCP server

Usually no published port is needed:

```bash
docker run --rm -i \
  -v "$HOME/mcp-data:/data" \
  IMAGE:TAG
```

## Day-2 commands

| Task | Command |
|---|---|
| Check Docker status | `systemctl --user status docker` |
| Restart Docker | `systemctl --user restart docker` |
| View Docker logs | `journalctl --user -u docker -n 200 --no-pager` |
| Confirm Docker socket | `echo $DOCKER_HOST` |
| Confirm storage driver | `docker info | grep 'Storage Driver'` |

## Final `daemon.json`

```json
{
  "features": {
    "containerd-snapshotter": false
  },
  "storage-driver": "fuse-overlayfs",
  "log-driver": "local"
}
```

## Summary

The final working procedure is:

1. install Docker from the official apt repository
2. install `uidmap`, `dbus-user-session`, `docker-ce-rootless-extras`, and `fuse-overlayfs`
3. add `subuid` and `subgid` ranges for the target user
4. disable the rootful Docker service and socket
5. enable lingering for the target user
6. run `dockerd-rootless-setuptool.sh install` as that user
7. export `DOCKER_HOST=unix:///run/user/$UID/docker.sock`
8. configure rootless Docker with:
   - `containerd-snapshotter: false`
   - `storage-driver: fuse-overlayfs`
   - `log-driver: local`
9. run `systemctl --user enable --now docker`
10. verify with `docker run --rm hello-world`

## Important note

Do **not** start Docker from `.bashrc`. The correct auto-start mechanism is:

- `systemctl --user enable docker`
- `loginctl enable-linger <user>`

That ensures Docker starts automatically after reboot.