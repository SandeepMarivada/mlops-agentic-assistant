#!/bin/bash
# ── EC2 Ubuntu Setup Script ───────────────────────────────────────────────────
# Run this ONCE on a fresh Ubuntu 22.04 EC2 instance before pushing to deploy.
#
# Usage:
#   chmod +x setup-ec2.sh
#   ./setup-ec2.sh
#
# After this script completes, log out and log back in, then verify:
#   docker run hello-world
#   docker compose version
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "=============================="
echo "  MLOps EC2 Setup — Ubuntu"
echo "=============================="

# ── 1. Update system ──────────────────────────────────────────────────────────
echo ""
echo "[1/5] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# ── 2. Install Git ────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Installing Git..."
sudo apt-get install -y git
git --version

# ── 3. Install Docker Engine (official Docker repo, not apt default) ──────────
echo ""
echo "[3/5] Installing Docker Engine..."
sudo apt-get install -y ca-certificates curl gnupg lsb-release

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
docker --version
docker compose version

# ── 4. Add ubuntu user to docker group ───────────────────────────────────────
echo ""
echo "[4/5] Adding ubuntu user to docker group..."
sudo usermod -aG docker ubuntu

# ── 5. Create /app directory ─────────────────────────────────────────────────
echo ""
echo "[5/5] Creating /app directory..."
sudo mkdir -p /app
sudo chown ubuntu:ubuntu /app

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "  Setup complete!"
echo "=============================="
echo ""
echo "IMPORTANT: You must log out and log back in for Docker"
echo "group permissions to take effect."
echo ""
echo "After re-login, verify with:"
echo "  docker run hello-world"
echo "  docker compose version"
echo ""
echo "Then add GitHub Secrets and push to deploy:"
echo "  EC2_HOST        = $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '<your-ec2-ip>')"
echo "  EC2_USERNAME    = ubuntu"
echo "  EC2_SSH_KEY     = <contents of your .pem file>"
echo "  OPENROUTER_API_KEY = <your openrouter key>"
