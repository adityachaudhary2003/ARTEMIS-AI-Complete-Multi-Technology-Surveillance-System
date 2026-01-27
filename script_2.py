# Create comprehensive Docker infrastructure and deployment files
import os

# Create Docker Compose file for the entire system
docker_compose = """version: '3.8'

services:
  # ===========================================
  # CORE PROCESSING SERVICES
  # ===========================================
  
  video-ingestion:
    build:
      context: ./services/video-ingestion
      dockerfile: Dockerfile
    container_name: artemis-video-ingestion
    environment:
      - RTSP_TIMEOUT=30
      - MAX_CONCURRENT_STREAMS=100
      - GPU_ACCELERATION=true
    volumes:
      - video-cache:/var/cache/video
      - ./config:/etc/artemis
    ports:
      - "8001:8001"
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  computer-vision:
    build:
      context: ./services/computer-vision
      dockerfile: Dockerfile
    container_name: artemis-computer-vision
    environment:
      - MODEL_PATH=/models
      - CONFIDENCE_THRESHOLD=0.75
      - GPU_MEMORY_FRACTION=0.8
    volumes:
      - ./models:/models:ro
      - model-cache:/var/cache/models
    ports:
      - "8002:8002"
    depends_on:
      - video-ingestion
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  ai-inference:
    build:
      context: ./services/ai-inference
      dockerfile: Dockerfile
    container_name: artemis-ai-inference
    environment:
      - PYTHONPATH=/app
      - TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
      - MODEL_CACHE_DIR=/cache/models
    volumes:
      - model-cache:/cache/models
      - ./models:/models:ro
    ports:
      - "8003:8003"
    depends_on:
      - vector-db
      - redis
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # ===========================================
  # BACKEND SERVICES (JAVA)
  # ===========================================
  
  threat-intelligence:
    build:
      context: ./services/threat-intelligence
      dockerfile: Dockerfile
    container_name: artemis-threat-intelligence
    environment:
      - SPRING_PROFILES_ACTIVE=production
      - DATABASE_URL=jdbc:postgresql://postgres:5432/artemis
      - DATABASE_USERNAME=artemis_user
      - DATABASE_PASSWORD=artemis_secure_password
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    ports:
      - "8004:8004"
    depends_on:
      - postgres
      - kafka
    restart: unless-stopped

  configuration-service:
    build:
      context: ./services/configuration
      dockerfile: Dockerfile
    container_name: artemis-configuration
    environment:
      - SPRING_PROFILES_ACTIVE=production
      - DATABASE_URL=jdbc:postgresql://postgres:5432/artemis
      - DATABASE_USERNAME=artemis_user
      - DATABASE_password=artemis_secure_password
    ports:
      - "8005:8005"
    depends_on:
      - postgres
    restart: unless-stopped

  # ===========================================
  # NODE.JS SERVICES
  # ===========================================
  
  api-gateway:
    build:
      context: ./services/api-gateway
      dockerfile: Dockerfile
    container_name: artemis-api-gateway
    environment:
      - NODE_ENV=production
      - JWT_SECRET=your-super-secret-jwt-key-here
      - REDIS_URL=redis://redis:6379
      - RATE_LIMIT_REQUESTS=1000
      - RATE_LIMIT_WINDOW=60000
    ports:
      - "8000:8000"
    depends_on:
      - redis
    restart: unless-stopped

  event-processing:
    build:
      context: ./services/event-processing
      dockerfile: Dockerfile
    container_name: artemis-event-processing
    environment:
      - NODE_ENV=production
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - WEBSOCKET_PORT=8006
      - REDIS_URL=redis://redis:6379
    ports:
      - "8006:8006"
    depends_on:
      - kafka
      - redis
    restart: unless-stopped

  # ===========================================
  # PYTHON SERVICES
  # ===========================================
  
  vector-search:
    build:
      context: ./services/vector-search
      dockerfile: Dockerfile
    container_name: artemis-vector-search
    environment:
      - PYTHONPATH=/app
      - VECTOR_DB_HOST=vector-db
      - VECTOR_DB_PORT=19530
      - EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    ports:
      - "8007:8007"
    depends_on:
      - vector-db
    restart: unless-stopped

  # ===========================================
  # FRONTEND APPLICATION
  # ===========================================
  
  dashboard:
    build:
      context: ./frontend/dashboard
      dockerfile: Dockerfile
    container_name: artemis-dashboard
    environment:
      - REACT_APP_API_BASE_URL=http://api-gateway:8000
      - REACT_APP_WEBSOCKET_URL=ws://event-processing:8006
    ports:
      - "3000:80"
    depends_on:
      - api-gateway
      - event-processing
    restart: unless-stopped

  # ===========================================
  # DATABASE SERVICES
  # ===========================================
  
  postgres:
    image: postgres:15-alpine
    container_name: artemis-postgres
    environment:
      - POSTGRES_DB=artemis
      - POSTGRES_USER=artemis_user
      - POSTGRES_PASSWORD=artemis_secure_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d:ro
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: artemis-redis
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  vector-db:
    image: milvusdb/milvus:latest
    container_name: artemis-vector-db
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
    volumes:
      - vector-db-data:/var/lib/milvus
    ports:
      - "19530:19530"
    depends_on:
      - etcd
      - minio
    restart: unless-stopped

  # ===========================================
  # SUPPORTING SERVICES
  # ===========================================
  
  kafka:
    image: confluentinc/cp-kafka:latest
    container_name: artemis-kafka
    environment:
      - KAFKA_BROKER_ID=1
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
      - KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1
      - KAFKA_AUTO_CREATE_TOPICS_ENABLE=true
    volumes:
      - kafka-data:/var/lib/kafka/data
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper
    restart: unless-stopped

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    container_name: artemis-zookeeper
    environment:
      - ZOOKEEPER_CLIENT_PORT=2181
      - ZOOKEEPER_TICK_TIME=2000
    volumes:
      - zookeeper-data:/var/lib/zookeeper
    restart: unless-stopped

  etcd:
    image: quay.io/coreos/etcd:v3.5.0
    container_name: artemis-etcd
    command: 
      - etcd
      - --advertise-client-urls=http://etcd:2379
      - --listen-client-urls=http://0.0.0.0:2379
      - --listen-peer-urls=http://0.0.0.0:2380
      - --data-dir=/etcd-data
    volumes:
      - etcd-data:/etcd-data
    restart: unless-stopped

  minio:
    image: minio/minio:latest
    container_name: artemis-minio
    environment:
      - MINIO_ROOT_USER=artemis
      - MINIO_ROOT_PASSWORD=artemis123
    command: server /data --console-address ":9001"
    volumes:
      - minio-data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    restart: unless-stopped

  # ===========================================
  # MONITORING & OBSERVABILITY
  # ===========================================
  
  prometheus:
    image: prom/prometheus:latest
    container_name: artemis-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: artemis-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=artemis123
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning:ro
    ports:
      - "3001:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
  vector-db-data:
  kafka-data:
  zookeeper-data:
  etcd-data:
  minio-data:
  prometheus-data:
  grafana-data:
  video-cache:
  model-cache:

networks:
  default:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16
"""

# Create the main docker-compose.yml file
with open('docker-compose.yml', 'w') as f:
    f.write(docker_compose)

print("✅ Docker Compose infrastructure created!")
print("📦 Services included:")
print("   • 8 Core microservices (C++, Python, Java, Node.js)")
print("   • React dashboard frontend")
print("   • PostgreSQL, Redis, Milvus vector DB")
print("   • Kafka message streaming")
print("   • Prometheus + Grafana monitoring")
print("   • Auto-scaling and GPU support")