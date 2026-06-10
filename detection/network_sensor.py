"""Optimized Network Sensor for AI-Powered SOC.

This script sniffs network traffic and extracts ONLY the 31 critical features
required to detect the 5 major CIC-DDoS-2018 attack types. By avoiding the full
70-feature calculation, this sensor is lightweight enough to run continuously.

Flows are tracked by 5-tuple (SrcIP, DstIP, SrcPort, DstPort, Protocol).
When a flow expires (or FIN/RST is seen), the 31 features are calculated
and pushed to Kafka as JSON.
"""

import json
import logging
import math
import socket
import time
from collections import defaultdict
from threading import Lock

from kafka import KafkaProducer
from scapy.all import IP, TCP, UDP, sniff

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("network_sensor")

FLOW_TIMEOUT = 10.0  # seconds to keep an idle flow before exporting

# Maps TCP flags to scapy representation
FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20
ECE = 0x40
CWR = 0x80  # CWE in CIC dataset

class Flow:
    """Tracks state for a single network flow to calculate the 31 features."""
    
    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        
        self.start_time = time.time()
        self.last_time = self.start_time
        
        # Packet counts and sizes
        self.fwd_packets = []  # sizes
        self.bwd_packets = []  # sizes
        self.fwd_timestamps = []
        self.bwd_timestamps = []
        
        self.fwd_header_len = 0
        self.bwd_header_len = 0
        
        self.fwd_psh_flags = 0
        
        self.flag_counts = {
            "ACK": 0, "CWE": 0, "RST": 0, "URG": 0
        }
        
        self.init_win_bytes_fwd = -1
        self.min_seg_size_fwd = -1
        
        self.is_closed = False

    def add_packet(self, pkt, direction, timestamp):
        """Add packet stats to flow."""
        self.last_time = timestamp
        
        if IP not in pkt:
            return
            
        ip_len = pkt[IP].len
        
        header_len = 20  # default IP
        if TCP in pkt:
            header_len = len(pkt[TCP])
            flags = pkt[TCP].flags
            if flags & ACK: self.flag_counts["ACK"] += 1
            if flags & CWR: self.flag_counts["CWE"] += 1
            if flags & RST: self.flag_counts["RST"] += 1
            if flags & URG: self.flag_counts["URG"] += 1
            
            if direction == "fwd":
                if self.init_win_bytes_fwd == -1:
                    self.init_win_bytes_fwd = pkt[TCP].window
                if flags & PSH:
                    self.fwd_psh_flags += 1
                
                # estimate min segment size
                if self.min_seg_size_fwd == -1 or header_len < self.min_seg_size_fwd:
                    self.min_seg_size_fwd = header_len
                    
            if flags & (FIN | RST):
                self.is_closed = True
                
        elif UDP in pkt:
            header_len = 8
            
        if direction == "fwd":
            self.fwd_packets.append(ip_len)
            self.fwd_timestamps.append(timestamp)
            self.fwd_header_len += header_len
        else:
            self.bwd_packets.append(ip_len)
            self.bwd_timestamps.append(timestamp)
            self.bwd_header_len += header_len

    def export(self):
        """Calculate the 31 required features and return JSON-ready dict."""
        duration = max(self.last_time - self.start_time, 0.0001)
        
        # Flow IATs
        all_timestamps = sorted(self.fwd_timestamps + self.bwd_timestamps)
        flow_iats = [all_timestamps[i] - all_timestamps[i-1] for i in range(1, len(all_timestamps))]
        
        # Fwd IATs
        fwd_iats = [self.fwd_timestamps[i] - self.fwd_timestamps[i-1] for i in range(1, len(self.fwd_timestamps))]
        
        # Bwd IATs
        bwd_iats = [self.bwd_timestamps[i] - self.bwd_timestamps[i-1] for i in range(1, len(self.bwd_timestamps))]
        
        all_pkts = self.fwd_packets + self.bwd_packets
        
        # Helper for stats
        def calc_mean(l): return sum(l)/len(l) if l else 0
        def calc_std(l, mean): return math.sqrt(sum((x-mean)**2 for x in l)/len(l)) if len(l) > 1 else 0

        fwd_mean = calc_mean(self.fwd_packets)
        flow_iat_mean = calc_mean(flow_iats)
        
        features = {
            # Metadata
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol_name": "TCP" if self.protocol == 6 else "UDP" if self.protocol == 17 else "OTHER",
            "timestamp": self.start_time,
            
            # 31 ML Features (mapped exactly to training pipeline columns)
            " ACK Flag Count": self.flag_counts["ACK"],
            " Average Packet Size": calc_mean(all_pkts),
            " Avg Fwd Segment Size": fwd_mean,
            " Bwd Header Length": self.bwd_header_len,
            " Bwd IAT Min": min(bwd_iats) * 1e6 if bwd_iats else 0,  # microseconds
            " Bwd IAT Total": sum(bwd_iats) * 1e6,
            " Bwd Packets/s": len(self.bwd_packets) / duration,
            " CWE Flag Count": self.flag_counts["CWE"],
            " Destination Port": self.dst_port,
            " Flow IAT Std": calc_std(flow_iats, flow_iat_mean) * 1e6,
            " Fwd Header Length": self.fwd_header_len,
            " Fwd IAT Mean": calc_mean(fwd_iats) * 1e6,
            " Fwd IAT Total": sum(fwd_iats) * 1e6,
            " Fwd PSH Flags": self.fwd_psh_flags,
            " Fwd Packet Length Max": max(self.fwd_packets) if self.fwd_packets else 0,
            " Fwd Packet Length Mean": fwd_mean,
            " Fwd Packet Length Min": min(self.fwd_packets) if self.fwd_packets else 0,
            " Fwd Packet Length Std": calc_std(self.fwd_packets, fwd_mean),
            " Inbound": 1 if self.dst_port in [80, 443, 22, 53, 3389] else 0, # rough heuristic
            " Init_Win_bytes_forward": self.init_win_bytes_fwd if self.init_win_bytes_fwd != -1 else 0,
            " Max Packet Length": max(all_pkts) if all_pkts else 0,
            " Min Packet Length": min(all_pkts) if all_pkts else 0,
            " Protocol": self.protocol,
            " RST Flag Count": self.flag_counts["RST"],
            " Source Port": self.src_port,
            " Subflow Bwd Packets": len(self.bwd_packets),
            " Subflow Fwd Packets": len(self.fwd_packets),
            " Total Fwd Packets": len(self.fwd_packets),
            " Total Length of Fwd Packets": sum(self.fwd_packets),
            " URG Flag Count": self.flag_counts["URG"],
            " min_seg_size_forward": self.min_seg_size_fwd if self.min_seg_size_fwd != -1 else 0,
        }
        return features


class NetworkSensor:
    def __init__(self, kafka_broker: str = "localhost:9092", topic: str = "network-flows"):
        self.kafka_broker = kafka_broker
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_broker],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.flows = {}
        self.lock = Lock()
        logger.info("Network Sensor initialized. Publishing to Kafka: %s", topic)

    def _get_flow_key(self, pkt):
        """Returns normalized flow key (directionless) and direction indicator."""
        if IP not in pkt:
            return None, None
            
        src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
        proto = pkt[IP].proto
        
        src_port, dst_port = 0, 0
        if TCP in pkt:
            src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport
        else:
            return None, None
            
        # Standardize key so a->b and b->a map to the same flow
        if (src_ip, src_port) < (dst_ip, dst_port):
            key = (src_ip, dst_ip, src_port, dst_port, proto)
            direction = "fwd"
        else:
            key = (dst_ip, src_ip, dst_port, src_port, proto)
            direction = "bwd"
            
        return key, direction

    def _packet_callback(self, pkt):
        key, direction = self._get_flow_key(pkt)
        if not key:
            return
            
        timestamp = time.time()
        
        with self.lock:
            if key not in self.flows:
                # new flow is initialized with the directionless key's tuple
                self.flows[key] = Flow(*key)
                
            flow = self.flows[key]
            flow.add_packet(pkt, direction, timestamp)
            
            if flow.is_closed:
                self._export_flow(key, flow)

    def _export_flow(self, key, flow):
        features = flow.export()
        self.producer.send(self.topic, features)
        del self.flows[key]
        logger.debug("Exported flow %s:%s -> %s:%s (pkts: %d)", 
                    flow.src_ip, flow.src_port, flow.dst_ip, flow.dst_port, 
                    features[" Total Fwd Packets"] + features[" Subflow Bwd Packets"])

    def _cleanup_loop(self):
        """Periodically export and remove idle flows."""
        while True:
            time.sleep(FLOW_TIMEOUT / 2)
            now = time.time()
            expired = []
            with self.lock:
                for key, flow in self.flows.items():
                    if now - flow.last_time > FLOW_TIMEOUT:
                        expired.append((key, flow))
                
                for key, flow in expired:
                    self._export_flow(key, flow)

    def start(self, interface=None):
        import threading
        cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        cleanup_thread.start()
        
        logger.info("Starting packet sniff on interface: %s", interface or "Default")
        sniff(iface=interface, prn=self._packet_callback, store=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interface", help="Network interface to sniff (e.g. eth0, Wi-Fi)")
    parser.add_argument("--broker", default="localhost:9092", help="Kafka broker")
    args = parser.parse_args()
    
    sensor = NetworkSensor(kafka_broker=args.broker)
    try:
        sensor.start(interface=args.interface)
    except KeyboardInterrupt:
        logger.info("Stopping Network Sensor.")
