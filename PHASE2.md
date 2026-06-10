# SOC Analyst Copilot - Phase 2: Kafka Streaming + Enrichment Pipeline

## Phase 2 Overview

Phase 2 builds on the Phase 1 Wazuh bridge to create a full streaming and enrichment pipeline:

1. **Wazuh bridge** (Phase 1) — authenticates with Wazuh API, polls alerts, normalizes them.
2. **Bridge-to-Kafka adapter** — consumes normalized alerts and produces them to Kafka `raw-alerts` topic.
3. **Enrichment pipeline** — consumes `raw-alerts`, applies four enrichment layers, produces `enriched-alerts`.

## Enrichment Layers

### 1. GeoIP Enrichment (`geoip.py`)
- Uses MaxMind GeoLite2-City database
- Resolves IP addresses to geographic coordinates
- Caches results in Redis (24-hour TTL)
- Skips private/reserved IPs gracefully
- Returns: country, city, latitude, longitude, is_tor, is_vpn

### 2. Threat Intelligence (`threat_intel.py`)
- Integrates with AbuseIPDB API
- Checks if an IP is known as a malicious attacker
- Respects rate limits (1000 requests/day free tier)
- Caches results in Redis (1-hour TTL)
- Returns: abuse_score (0-100), is_known_attacker (bool), last_reported

### 3. MITRE Mapping (`mitre_mapper.py`)
- Maps Wazuh rule groups to MITRE ATT&CK techniques
- Configuration in `data/mitre_rule_mapping.yaml`
- Starter mappings: sshd→T1110, web→T1190, fim→T1565, syslog→T1078
- Returns: list of technique objects with technique_id, technique_name, tactic

### 4. Session Tracking (`sessionizer.py`)
- Tracks user sessions in Redis sorted sets
- Groups events by source IP within a 5-minute window
- Creates new session for each IP or reuses active session
- Returns: session_id, session_event_count, session_duration_seconds

## Enriched Alert Shape

Base normalized fields + enrichment fields:
```python
{
    # Phase 1 normalized fields
    "id": str,
    "timestamp": datetime,
    "rule_level": int,
    "rule_description": str,
    "rule_groups": list[str],
    "agent_id": str,
    "agent_name": str,
    "agent_ip": str,
    "src_ip": str,
    "dst_ip": str,
    "src_port": int,
    "protocol": str,

    # Phase 2 enrichment fields
    "geoip": {
        "src": {country, city, lat, lon, is_tor, is_vpn},
        "agent": {country, city, lat, lon, is_tor, is_vpn}
    },
    "threat_intel": {
        "src": {abuse_score, is_known_attacker, last_reported}
    },
    "mitre_techniques": [
        {technique_id, technique_name, tactic}
    ],
    "session_id": str,
    "session_event_count": int,
    "session_duration_seconds": int,
    "asset_criticality": str
}
```

## Infrastructure

### Kafka (Single-Broker Setup)
- Broker: `kafka:29092` (internal), `localhost:9092` (external)
- Topics: `raw-alerts`, `enriched-alerts`, `detections` (3 partitions each)
- Zookeeper coordination

### Redis
- Caching: GeoIP, threat intelligence, daily API call counters
- Session tracking: sorted sets by source IP with 1-hour TTL
- Port: `6379`

## Running Phase 2

### 1. Start Kafka + Redis
```bash
cd infra
docker-compose -f docker-compose.kafka.yml up -d
```

### 2. Start the Wazuh Bridge (Phase 1)
```bash
python consumer.py
# or if using Kafka producer:
python bridge_to_kafka.py
```

### 3. Start the Enrichment Pipeline
```bash
python pipeline/kafka_consumer.py
```

### Monitoring

View raw alerts being produced to Kafka:
```bash
docker exec -it wazuh-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:29092 \
  --topic raw-alerts \
  --from-beginning
```

View enriched alerts after enrichment:
```bash
docker exec -it wazuh-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:29092 \
  --topic enriched-alerts \
  --from-beginning
```

## Configuration

Copy `.env.example` to `.env` and set:
- `KAFKA_BROKERS` — Kafka broker addresses
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` — Redis connection
- `ABUSEIPDB_API_KEY` — AbuseIPDB API key (optional; threat intel will skip if empty)
- `GEOIP_DATABASE_PATH` — Path to GeoLite2-City.mmdb (download from MaxMind)

## Testing

Run the enrichment module tests:
```bash
python -m pytest tests/test_enrichment.py -v
```

## Files

- `infra/docker-compose.kafka.yml` — Kafka + Zookeeper + Redis
- `infra/kafka-init.sh` — Topic creation script
- `pipeline/kafka_producer.py` — Producer for raw alerts
- `pipeline/kafka_consumer.py` — Enrichment orchestrator
- `pipeline/kafka_utils.py` — Serialization and constants
- `pipeline/enrichment/geoip.py` — GeoIP enricher
- `pipeline/enrichment/threat_intel.py` — AbuseIPDB enricher
- `pipeline/enrichment/mitre_mapper.py` — MITRE mapper
- `pipeline/enrichment/sessionizer.py` — Session tracker
- `bridge_to_kafka.py` — Wazuh bridge → Kafka adapter
- `data/mitre_rule_mapping.yaml` — MITRE rule mappings
