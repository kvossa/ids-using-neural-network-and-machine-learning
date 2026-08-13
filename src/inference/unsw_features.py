import time
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

FLOW_TIMEOUT = 10.0

TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src", "ip.dst",
    "tcp.srcport", "tcp.dstport",
    "udp.srcport", "udp.dstport",
    "ip.proto",
    "tcp.flags",
    "tcp.window_size",
    "ip.ttl",
    "frame.len",
    "tcp.seq",
    "tcp.ack",
    "tcp.nxtseq",
    "tcp.options.timestamp.tsval",
]


_TCP_FLAGS_HEX = {
    0x01: "F", 0x02: "S", 0x04: "R", 0x08: "P",
    0x10: "A", 0x20: "U", 0x40: "E", 0x80: "C",
}

IP_PROTO_MAP = {
    0: "hopopt", 1: "icmp", 2: "igmp", 3: "ggp", 4: "ipv4", 5: "st",
    6: "tcp", 7: "cbt", 8: "egp", 9: "igp", 10: "bbn-rcc-mon",
    11: "nvp-ii", 12: "pup", 14: "emcon", 15: "xnet", 16: "chaos",
    17: "udp", 18: "mux", 19: "dcn-meas", 20: "hmp", 21: "prm",
    22: "xns-idp", 23: "trunk-1", 24: "trunk-2", 25: "leaf-1",
    26: "leaf-2", 27: "rdp", 28: "irtp", 29: "iso-tp4", 30: "netblt",
    31: "mfe-nsp", 32: "merit-inp", 33: "dccp", 34: "3pc", 35: "idpr",
    36: "xtp", 37: "ddp", 38: "idpr-cmtp", 39: "tp++", 40: "il",
    41: "ipv6", 42: "sdrp", 43: "ipv6-route", 44: "ipv6-frag",
    45: "idrp", 46: "rsvp", 47: "gre", 48: "dsr", 49: "bna",
    50: "esp", 51: "ah", 52: "i-nlsp", 54: "narp", 55: "min-ipv4",
    56: "tlsp", 57: "skip", 58: "ipv6-icmp", 59: "ipv6-nonxt",
    60: "ipv6-opts", 62: "cftp", 64: "sat-expak", 65: "kryptolan",
    66: "rvd", 67: "ippc", 69: "sat-mon", 70: "visa", 71: "ipcv",
    72: "cpnx", 73: "cphb", 74: "wsn", 75: "pvp", 76: "br-sat-mon",
    77: "sun-nd", 78: "wb-mon", 79: "wb-expak", 80: "iso-ip",
    81: "vmtp", 82: "secure-vmtp", 83: "vines", 84: "iptm",
    85: "nsfnet-igp", 86: "dgp", 87: "tcf", 88: "eigrp",
    89: "ospfigp", 90: "sprite-rpc", 91: "larp", 92: "mtp",
    93: "ax.25", 94: "ipip", 96: "scc-sp", 97: "etherip", 98: "encap",
    100: "gmtp", 101: "ifmp", 102: "pnni", 103: "pim", 104: "aris",
    105: "scps", 106: "qnx", 107: "a/n", 108: "ipcomp", 109: "snp",
    110: "compaq-peer", 111: "ipx-in-ip", 112: "vrrp", 113: "pgm",
    115: "l2tp", 116: "ddx", 117: "iatp", 118: "stp", 119: "srp",
    120: "uti", 121: "smp", 123: "ptp", 125: "fire", 126: "crtp",
    127: "crudp", 128: "sscopmce", 129: "iplt", 130: "sps",
    131: "pipe", 132: "sctp", 133: "fc", 134: "rsvp-e2e-ignore",
    136: "udplite", 137: "mpls-in-ip", 138: "manet", 139: "hip",
    140: "shim6", 141: "wesp", 142: "rohc", 143: "ethernet",
    144: "aggfrag", 145: "nsh", 146: "homa", 147: "bit-emu",
    255: "reserved",
}

SERVICE_MAP = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet",
    25: "smtp", 53: "dns", 67: "dhcp", 68: "dhcp",
    80: "http", 110: "pop3", 123: "ntp", 143: "imap",
    161: "snmp", 162: "snmp", 194: "irc", 389: "ldap",
    443: "ssl", 993: "ssl", 995: "ssl",
    1812: "radius", 1813: "radius", 3306: "mysql",
    5432: "postgresql", 6379: "redis", 8080: "http",
    8443: "ssl",
}


def _detect_service(pkt: Dict) -> str:
    for port in (pkt.get("dst_port", 0), pkt.get("src_port", 0)):
        svc = SERVICE_MAP.get(port)
        if svc:
            return svc
    return "-"


def _flags_to_str(hex_str: str) -> str:
    try:
        val = int(hex_str, 16) if hex_str else 0
    except ValueError:
        return ""
    return "".join(c for bit, c in _TCP_FLAGS_HEX.items() if val & bit)


def _parse_tshark_line(line: str) -> Optional[Dict]:
    parts = line.strip().split("\t")
    if len(parts) < 9:
        return None
    try:
        ts = float(parts[0]) if parts[0] else 0
    except ValueError:
        return None
    src_ip = parts[1] or "0.0.0.0"
    dst_ip = parts[2] or "0.0.0.0"
    src_port = parts[3] or parts[5] or "0"
    dst_port = parts[4] or parts[6] or "0"
    try:
        src_port = int(src_port)
        dst_port = int(dst_port)
    except ValueError:
        return None
    try:
        proto = int(parts[7]) if parts[7] else 0
    except ValueError:
        proto = 0
    tcp_flags = _flags_to_str(parts[8])
    try:
        window = int(parts[9]) if parts[9] else 0
    except ValueError:
        window = 0
    try:
        ttl = int(parts[10]) if parts[10] else 255
    except ValueError:
        ttl = 255
    try:
        length = int(parts[11]) if parts[11] else 0
    except ValueError:
        length = 0
    tcp_seq = parts[12] if len(parts) > 12 else ""
    tcp_ack = parts[13] if len(parts) > 13 else ""
    tcp_nxtseq = parts[14] if len(parts) > 14 else ""
    return {
        "time": ts,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "proto": proto,
        "tcp_flags": tcp_flags,
        "window": window,
        "ttl": ttl,
        "length": length,
        "tcp_seq": tcp_seq,
        "tcp_ack": tcp_ack,
        "tcp_nxtseq": tcp_nxtseq,
    }


def _is_forward(pkt: Dict, flow_key: Tuple) -> bool:
    return (pkt["src_ip"], pkt["dst_ip"], pkt["src_port"], pkt["dst_port"], pkt["proto"]) == flow_key


class UNSWFlow:
    def __init__(self, pkt: Dict):
        fwd = (
            pkt["src_ip"], pkt["dst_ip"],
            pkt["src_port"], pkt["dst_port"],
            pkt["proto"],
        )
        rev = (
            pkt["dst_ip"], pkt["src_ip"],
            pkt["dst_port"], pkt["src_port"],
            pkt["proto"],
        )
        self.flow_key = fwd
        self.rev_key = rev
        self.src_ip = pkt["src_ip"]
        self.dst_ip = pkt["dst_ip"]
        self.src_port = pkt["src_port"]
        self.dst_port = pkt["dst_port"]
        self.proto = pkt["proto"]
        self.proto_str = IP_PROTO_MAP.get(pkt["proto"], str(pkt["proto"]))
        self._is_forward = True

        self.start_time = pkt["time"]
        self.last_time = pkt["time"]

        self.spkts = 1 if self._is_forward else 0
        self.dpkts = 0 if self._is_forward else 1
        self.sbytes = pkt["length"] if self._is_forward else 0
        self.dbytes = 0 if self._is_forward else pkt["length"]
        self.sttl = pkt["ttl"]
        self.dttl = 255
        self.sttl_list = [pkt["ttl"]]
        self.dttl_list = []
        self.sinpkt_vals = []
        self.dinpkt_vals = []
        self.last_fwd_time = pkt["time"] if self._is_forward else None
        self.last_rev_time = None if self._is_forward else pkt["time"]
        self.swin = pkt["window"] if self._is_forward else 0
        self.dwin = 0 if self._is_forward else pkt["window"]
        self.syn_seen = "S" in pkt["tcp_flags"]
        self.synack_seen = False
        self.ack_seen = False
        self.fin_seen = False
        self.rst_seen = False
        self.syn_time = pkt["time"] if self.syn_seen else None
        self.synack_time = None
        self.ack_data_time = None
        self.retrans_fwd = 0
        self.retrans_rev = 0
        self.fwd_seq_max = 0
        self.rev_seq_max = 0
        self.max_sbytes = pkt["length"]
        self.max_dbytes = 0
        self.sum_sbytes = pkt["length"]
        self.sum_dbytes = 0
        self.sload = 0.0
        self.dload = 0.0
        self.smean = float(pkt["length"])
        self.dmean = 0.0
        self.service = _detect_service(pkt)
        self.state = "INT"
        self.trans_depth = 0
        self.response_body_len = 0
        self.is_ftp_login = 0
        self.is_sm_ips_ports = 0
        self.dur = 0.0

    def add_packet(self, pkt: Dict):
        fwd = self._is_forward = (pkt["src_ip"], pkt["dst_ip"], pkt["src_port"], pkt["dst_port"], pkt["proto"]) == self.flow_key
        now = pkt["time"]
        dt = now - self.last_time if self.last_time else 0
        self.last_time = now
        flags = pkt["tcp_flags"]

        if fwd:
            self.spkts += 1
            self.sbytes += pkt["length"]
            self.sum_sbytes += pkt["length"]
            if pkt["length"] > self.max_sbytes:
                self.max_sbytes = pkt["length"]
            self.sttl_list.append(pkt["ttl"])
            self.sttl = min(self.sttl, pkt["ttl"])
            self.swin = max(self.swin, pkt["window"]) if pkt["window"] else self.swin
            if self.last_fwd_time is not None and dt > 0:
                self.sinpkt_vals.append(dt)
            self.last_fwd_time = now

            if not self.syn_seen and "S" in flags:
                self.syn_seen = True
                self.syn_time = now
                self.state = "REQ"
            if "F" in flags:
                self.fin_seen = True
            if "R" in flags:
                self.rst_seen = True
        else:
            self.dpkts += 1
            self.dbytes += pkt["length"]
            self.sum_dbytes += pkt["length"]
            if pkt["length"] > self.max_dbytes:
                self.max_dbytes = pkt["length"]
            self.dttl_list.append(pkt["ttl"])
            self.dttl = min(self.dttl, pkt["ttl"])
            self.dwin = max(self.dwin, pkt["window"]) if pkt["window"] else self.dwin
            if self.last_rev_time is not None and dt > 0:
                self.dinpkt_vals.append(dt)
            self.last_rev_time = now

            if "S" in flags and "A" in flags and self.syn_seen and not self.synack_seen:
                self.synack_seen = True
                self.synack_time = now
                self.state = "ACC"
            if "A" in flags:
                self.ack_seen = True
                if self.synack_seen and self.ack_data_time is None:
                    self.ack_data_time = now
                    self.state = "CON"
            if "F" in flags:
                self.fin_seen = True
                self.state = "FIN"
            if "R" in flags:
                self.rst_seen = True
                self.state = "RST"

        self.dur = self.last_time - self.start_time
        if self.dur > 0:
            self.sload = (self.sbytes * 8.0) / self.dur
            self.dload = (self.dbytes * 8.0) / self.dur
        if self.spkts > 0:
            self.smean = self.sum_sbytes / float(self.spkts)
        if self.dpkts > 0:
            self.dmean = self.sum_dbytes / float(self.dpkts)

    def is_completed(self, now: float) -> bool:
        return self.fin_seen or self.rst_seen or (now - self.last_time > FLOW_TIMEOUT)

    def to_dict(self) -> Dict:
        dur = max(self.dur, 0.000001)
        rate = self.spkts / dur if dur > 0 else 0
        sinpkt = sum(self.sinpkt_vals) / len(self.sinpkt_vals) if self.sinpkt_vals else 0
        dinpkt = sum(self.dinpkt_vals) / len(self.dinpkt_vals) if self.dinpkt_vals else 0
        sjit = 0.0
        djit = 0.0
        if len(self.sinpkt_vals) > 1:
            sjit = sum(abs(self.sinpkt_vals[i] - self.sinpkt_vals[i-1]) for i in range(1, len(self.sinpkt_vals))) / (len(self.sinpkt_vals) - 1)
        if len(self.dinpkt_vals) > 1:
            djit = sum(abs(self.dinpkt_vals[i] - self.dinpkt_vals[i-1]) for i in range(1, len(self.dinpkt_vals))) / (len(self.dinpkt_vals) - 1)

        tcprtt = 0.0
        synack = 0.0
        ackdat = 0.0
        if self.syn_time and self.synack_time:
            synack = self.synack_time - self.syn_time
        if self.synack_time and self.ack_data_time:
            ackdat = self.ack_data_time - self.synack_time
        tcprtt = synack + ackdat

        sttl = min(self.sttl_list) if self.sttl_list else 255
        dttl = min(self.dttl_list) if self.dttl_list else 255

        return {
            "dur": round(dur, 6),
            "proto": self.proto_str,
            "service": self.service,
            "state": self.state,
            "spkts": self.spkts,
            "dpkts": self.dpkts,
            "sbytes": self.sbytes,
            "dbytes": self.dbytes,
            "rate": round(rate, 6),
            "sttl": sttl,
            "dttl": dttl,
            "sload": round(self.sload, 6),
            "dload": round(self.dload, 6),
            "sloss": 0,
            "dloss": 0,
            "sinpkt": round(sinpkt, 6),
            "dinpkt": round(dinpkt, 6),
            "sjit": round(sjit, 6),
            "djit": round(djit, 6),
            "swin": self.swin,
            "stcpb": 0,
            "dtcpb": 0,
            "dwin": self.dwin,
            "tcprtt": round(tcprtt, 6),
            "synack": round(synack, 6),
            "ackdat": round(ackdat, 6),
            "smean": round(self.smean, 6),
            "dmean": round(self.dmean, 6),
            "trans_depth": self.trans_depth,
            "response_body_len": self.response_body_len,
            "ct_srv_src": 0,
            "ct_state_ttl": 0,
            "ct_dst_ltm": 0,
            "ct_src_dport_ltm": 0,
            "ct_dst_sport_ltm": 0,
            "ct_dst_src_ltm": 0,
            "is_ftp_login": self.is_ftp_login,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0,
            "ct_src_ltm": 0,
            "ct_srv_dst": 0,
            "is_sm_ips_ports": self.is_sm_ips_ports,
        }


class ConnectionStateTable:
    def __init__(self, max_connections: int = 1000):
        self.completed_flows: deque = deque(maxlen=max_connections)
        self.ct_srv_src = defaultdict(int)
        self.ct_state_ttl = defaultdict(int)
        self.ct_dst_ltm = defaultdict(int)
        self.ct_src_dport_ltm = defaultdict(int)
        self.ct_dst_sport_ltm = defaultdict(int)
        self.ct_dst_src_ltm = defaultdict(int)
        self.ct_src_ltm = defaultdict(int)
        self.ct_srv_dst = defaultdict(int)

    def reset(self):
        for attr in ['ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
                     'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                     'ct_dst_src_ltm', 'ct_src_ltm', 'ct_srv_dst']:
            getattr(self, attr).clear()

    def add_flow(self, flow: UNSWFlow, features: Dict):
        self.completed_flows.append((flow, features))
        state = features["state"]
        sttl = features["sttl"]
        dttl = features["dttl"]
        service = features["service"]
        src = flow.src_ip
        dst = flow.dst_ip
        sport = flow.src_port
        dport = flow.dst_port
        self.ct_srv_src[(src, service)] += 1
        self.ct_state_ttl[(state, sttl, dttl)] += 1
        self.ct_dst_ltm[dst] += 1
        self.ct_src_dport_ltm[(src, dport)] += 1
        self.ct_dst_sport_ltm[(dst, sport)] += 1
        self.ct_dst_src_ltm[(src, dst)] += 1
        self.ct_src_ltm[src] += 1
        self.ct_srv_dst[(service, dst)] += 1

    def update_features(self, flow: UNSWFlow, features: Dict):
        src = flow.src_ip
        dst = flow.dst_ip
        sport = flow.src_port
        dport = flow.dst_port
        service = features["service"]
        state = features["state"]
        sttl = features["sttl"]
        dttl = features["dttl"]
        features["ct_srv_src"] = self.ct_srv_src.get((src, service), 0)
        features["ct_state_ttl"] = self.ct_state_ttl.get((state, sttl, dttl), 0)
        features["ct_dst_ltm"] = self.ct_dst_ltm.get(dst, 0)
        features["ct_src_dport_ltm"] = self.ct_src_dport_ltm.get((src, dport), 0)
        features["ct_dst_sport_ltm"] = self.ct_dst_sport_ltm.get((dst, sport), 0)
        features["ct_dst_src_ltm"] = self.ct_dst_src_ltm.get((src, dst), 0)
        features["ct_src_ltm"] = self.ct_src_ltm.get(src, 0)
        features["ct_srv_dst"] = self.ct_srv_dst.get((service, dst), 0)
