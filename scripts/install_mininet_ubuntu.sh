#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this installer with sudo." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y mininet openvswitch-switch iperf3 iproute2 iputils-ping
service openvswitch-switch start

mn --version
python3 -c "import mininet; print('Mininet Python API ready')"
mn --switch ovsbr --test pingall
mn -c
