# Lab record — FICHA_LABORATORIO

Canonical description of the IDS lab VM environment and a per-session template.

## 1. Lab environment (topology)

Four-node lab on **KVM/libvirt**. The IDS node is the host machine that runs capture and inference; the other three are Ubuntu VMs.

| Node | OS | Role |
|------|----|------|
| Client | Ubuntu (VM) | Generates benign traffic |
| Server | Ubuntu (VM) | Target — runs HTTP + SSH |
| Attacker | Ubuntu (VM) | Generates malicious traffic |
| IDS node | Manjaro (host) | Capture + inference (`infer_live.py`) |

Network: private lab subnet **192.168.100.0/24**.

### Capture topologies

Two ways to feed traffic to the IDS node:

- **VM experiment** — the IDS node listens on the isolated network where the three VMs operate (promiscuous/bridge on the KVM network).
- **Physical lab experiment** — a SPAN/mirror port on the switch forwards the mirror to the IDS capture interface.

The capture interface is configured in `config/lab.json` / `config/lab.yaml` (`capture_interface`).

## 2. Virtualization

| Item | Value |
|------|-------|
| Hypervisor | KVM/libvirt |
| VM guests | Client, Server, Attacker (Ubuntu) |
| Host (IDS node) | Manjaro |
| VM network | Isolated 192.168.100.0/24 (see topology) |

## 3. Services on the Server node

Enable before a session; without these the scenarios are not reproducible.

- [ ] HTTP server on port 80 (test URL: `http://192.168.100.X/`)
- [ ] OpenSSH server on port 22
- [ ] Firewall rules for ports 80/22 (test segment only)
- [ ] Test accounts (lab-only users, no real domain)

## 4. IP addressing

| Node | IPv4 | Notes |
|------|------|-------|
| Client | `192.168.100.10` | |
| Attacker | `192.168.100.20` | |
| Server | `192.168.100.30` | |
| IDS node | `192.168.100.40` | capture interface |

Fill the actual values in `FICHA_LABORATORIO.local.md`.

## 5. Experiment catalog

| Experiment | Script | Direction | Tool | Target |
|------------|--------|-----------|------|--------|
| HTTP benign | `scripts/normal/http_benign.sh` | Client → Server | `curl` | :80 |
| SSH benign | manual (no script yet) | Client → Server | `ssh` | :22 |
| Port scan | `scripts/malicious/port_scan.sh` | Attacker → Server | `nmap -sS` | :22,80 |
| Brute force | `scripts/malicious/brute_force_ssh.sh` (stub) | Attacker → Server | `hydra` | :22 |
| DoS | `scripts/malicious/dos_hping.sh` (stub) | Attacker → Server | `hping3` | :80 |

Notes:

- `brute_force_ssh.sh` and `dos_hping.sh` are **templates that currently send no traffic** — edit and obtain explicit authorization before running.
- There is no SSH-benign script yet; SSH benign traffic was generated manually.
- Attack scenarios require authorization and are restricted to the isolated 192.168.100.0/24 segment.

## 6. Session record

| Field | Value |
|-------|-------|
| Date | |
| Responsible | |
| Repo commit (`git rev-parse --short HEAD`) | |
| Preprocessing artifact (`preprocessing.pkl`) | |
| CIC stage 1 (`stage1.keras`) | |
| CIC stage 2 (`stage2.keras` + `threshold.json`) | |
| Fine-tuned head (`models/classification/fine_tuned/`, if any) | |
| Capture interface | |
| `infer_live.py` command | |
| Scenarios run | |
| Predictions CSV / GT CSV | |
| Metrics (`evaluate_lab.py` JSON): F1, benign FP rate, latency | |

## 7. Quick verification (on the IDS node)

```bash
ip -br a
sudo tcpdump -i <CAPTURE_IFACE> -c 5 -n
ping -c 2 <SERVER_IP>
ip route
```
