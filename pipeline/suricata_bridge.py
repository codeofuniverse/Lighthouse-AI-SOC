"""Suricata to Kafka Bridge.

Reads Suricata's eve.json log file, filters for 'flow' events,
maps the Suricata flow statistics to the 13 CIC-compatible features
expected by the ML Classifier, and pushes to Kafka.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict

from kafka import KafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("suricata_bridge")

# The exact 13 features the ML model expects
CIC_FEATURES = [
    " Destination Port",
    " Protocol",
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    " FIN Flag Count",
    " SYN Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " ACK Flag Count",
    " URG Flag Count",
]


def _map_eve_to_cic(eve_flow: Dict[str, Any]) -> Dict[str, Any]:
    """Maps a Suricata EVE flow event to CIC-DDoS features."""
    
    # 1. Base network info
    dst_port = eve_flow.get("dest_port", 0)
    proto_str = eve_flow.get("proto", "TCP").upper()
    protocol = 6 if proto_str == "TCP" else 17 if proto_str == "UDP" else 0
    
    flow_data = eve_flow.get("flow", {})
    tcp_data = eve_flow.get("tcp", {})

    # 2. Flow stats mapping
    total_fwd_packets = flow_data.get("pkts_toserver", 0)
    total_bwd_packets = flow_data.get("pkts_toclient", 0)
    total_len_fwd = flow_data.get("bytes_toserver", 0)
    total_len_bwd = flow_data.get("bytes_toclient", 0)
    
    # Duration (Suricata age is in seconds, CIC Flow Duration is typically microseconds)
    age_seconds = flow_data.get("age", 0)
    flow_duration_us = int(age_seconds * 1_000_000)

    # 3. TCP Flags mapping (booleans in Suricata)
    syn_count = 1 if tcp_data.get("syn") else 0
    ack_count = 1 if tcp_data.get("ack") else 0
    psh_count = 1 if tcp_data.get("psh") else 0
    fin_count = 1 if tcp_data.get("fin") else 0
    rst_count = 1 if tcp_data.get("rst") else 0
    urg_count = 1 if tcp_data.get("urg") else 0

    return {
        "src_ip": eve_flow.get("src_ip", ""),
        "dst_ip": eve_flow.get("dest_ip", ""),
        "src_port": eve_flow.get("src_port", 0),
        
        # ML mapped features
        " Destination Port": dst_port,
        " Protocol": protocol,
        " Flow Duration": flow_duration_us,
        " Total Fwd Packets": total_fwd_packets,
        " Total Backward Packets": total_bwd_packets,
        "Total Length of Fwd Packets": total_len_fwd,
        " Total Length of Bwd Packets": total_len_bwd,
        " FIN Flag Count": fin_count,
        " SYN Flag Count": syn_count,
        " RST Flag Count": rst_count,
        " PSH Flag Count": psh_count,
        " ACK Flag Count": ack_count,
        " URG Flag Count": urg_count,
    }


def tail_eve_json(file_path: str, producer: KafkaProducer, topic: str):
    """Tails the eve.json file similar to 'tail -f'."""
    
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} does not exist. Waiting for it to be created...")
        while not os.path.exists(file_path):
            time.sleep(1)
            
    logger.info(f"Tailing Suricata logs from {file_path}")
    
    with open(file_path, 'r') as f:
        # Go to the end of file
        f.seek(0, os.SEEK_END)
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
                
            try:
                event = json.loads(line)
                
                # We only care about flow records for ML
                if event.get("event_type") == "flow":
                    mapped_features = _map_eve_to_cic(event)
                    producer.send(topic, mapped_features)
                    logger.debug(f"Pushed mapped flow to {topic}: {mapped_features['src_ip']} -> {event.get('dest_ip')}")
                    
            except json.JSONDecodeError:
                pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eve-log", default="data/suricata/eve.json", help="Path to Suricata eve.json")
    parser.add_argument("--broker", default="localhost:9092", help="Kafka broker")
    parser.add_argument("--topic", default="network-flows", help="Kafka topic for ML ingest")
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=[args.broker],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    
    try:
        tail_eve_json(args.eve_log, producer, args.topic)
    except KeyboardInterrupt:
        logger.info("Stopping Suricata Bridge.")
