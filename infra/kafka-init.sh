#!/bin/bash
set -e

# Wait for Kafka to be ready
until /opt/bitnami/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server kafka:29092; do
  echo "Waiting for Kafka broker..."
  sleep 1
done

# Create topics
/opt/bitnami/kafka/bin/kafka-topics.sh \
  --create \
  --bootstrap-server kafka:29092 \
  --topic raw-alerts \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists

/opt/bitnami/kafka/bin/kafka-topics.sh \
  --create \
  --bootstrap-server kafka:29092 \
  --topic enriched-alerts \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists

/opt/bitnami/kafka/bin/kafka-topics.sh \
  --create \
  --bootstrap-server kafka:29092 \
  --topic detections \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists

echo "Topics created successfully"
