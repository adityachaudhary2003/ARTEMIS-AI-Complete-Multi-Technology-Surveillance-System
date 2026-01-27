# ARTEMIS AI Surveillance System - Deployment Guide

## System Requirements

### Hardware Requirements
- **CPU**: Intel Xeon or AMD EPYC (min 16 cores)
- **RAM**: 64GB minimum, 128GB recommended
- **GPU**: NVIDIA RTX 4090 or Tesla A100 (multiple GPUs recommended)
- **Storage**: 
  - 2TB NVMe SSD for application data
  - 10TB+ for video storage and model cache
- **Network**: 10Gbps network interface for high-throughput video streaming

### Software Requirements
- **OS**: Ubuntu 20.04 LTS or CentOS 8
- **Docker**: 24.0+ with NVIDIA Container Toolkit
- **Docker Compose**: v2.20+
- **NVIDIA Drivers**: 530+ with CUDA 11.8+

## Pre-deployment Setup

### 1. Install NVIDIA Container Toolkit
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker
```

### 2. Configure System Resources
```bash
# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Configure kernel parameters for high-performance networking
echo "net.core.rmem_max = 134217728" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max = 134217728" | sudo tee -a /etc/sysctl.conf
echo "net.ipv4.tcp_rmem = 4096 87380 134217728" | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p
```

### 3. Create Directory Structure
```bash
# Create project directory
mkdir -p /opt/artemis-ai
cd /opt/artemis-ai

# Create data directories
mkdir -p data/{postgres,redis,milvus,kafka,prometheus,grafana}
mkdir -p logs/{services,system}
mkdir -p models/{yolo,face-recognition,behavior-analysis}
mkdir -p config/{prometheus,grafana,nginx}

# Set proper permissions
sudo chown -R $USER:$USER /opt/artemis-ai
chmod -R 755 /opt/artemis-ai
```

## Deployment Steps

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/artemis-ai-system.git /opt/artemis-ai
cd /opt/artemis-ai
```

### 2. Configure Environment Variables
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 3. Download Pre-trained Models
```bash
# Download AI models (this may take some time)
./scripts/download_models.sh
```

### 4. Build and Start Services
```bash
# Build all services
docker-compose build

# Start infrastructure services first
docker-compose up -d postgres redis vector-db kafka zookeeper etcd minio

# Wait for infrastructure to be ready (check logs)
docker-compose logs -f postgres redis vector-db

# Start core processing services
docker-compose up -d video-ingestion computer-vision ai-inference

# Start backend services
docker-compose up -d threat-intelligence configuration-service

# Start API and frontend services
docker-compose up -d api-gateway event-processing dashboard

# Start monitoring
docker-compose up -d prometheus grafana
```

### 5. Verify Deployment
```bash
# Check service health
curl http://localhost:8000/api/system/health

# Check individual services
docker-compose ps
docker-compose logs api-gateway
```

## Configuration

### Camera Integration
```bash
# Add RTSP camera streams
curl -X POST http://localhost:8000/api/v1/config/cameras \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "camera_id": "CAM-001",
    "name": "Main Entrance",
    "rtsp_url": "rtsp://camera-ip:554/stream",
    "location": "Building A - Lobby",
    "enabled": true
  }'
```

### AI Model Configuration
```bash
# Configure detection thresholds
curl -X PUT http://localhost:8000/api/v1/config/ai-models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "weapon_detection_threshold": 0.75,
    "face_recognition_threshold": 0.8,
    "behavior_anomaly_threshold": 0.65
  }'
```

## Scaling and Performance

### Horizontal Scaling
```bash
# Scale specific services
docker-compose up -d --scale ai-inference=3
docker-compose up -d --scale computer-vision=2
```

### GPU Resource Management
```bash
# Monitor GPU usage
nvidia-smi
watch -n 1 nvidia-smi

# Configure GPU allocation in docker-compose.yml
# Each service can specify GPU requirements
```

### Database Optimization
```bash
# PostgreSQL tuning for high-throughput
# Edit postgresql.conf:
shared_buffers = 8GB
effective_cache_size = 24GB
max_connections = 200
work_mem = 256MB
```

## Monitoring and Maintenance

### Access Monitoring Dashboards
- **Grafana**: http://localhost:3001 (admin/artemis123)
- **Prometheus**: http://localhost:9090
- **Artemis Dashboard**: http://localhost:3000

### Log Management
```bash
# View service logs
docker-compose logs -f service-name

# Centralized logging with ELK stack (optional)
docker-compose -f docker-compose.elk.yml up -d
```

### Backup Procedures
```bash
# Backup databases
./scripts/backup_postgres.sh
./scripts/backup_vector_db.sh

# Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz config/
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
   ```

2. **High Memory Usage**
   ```bash
   # Monitor memory usage
   docker stats
   # Adjust memory limits in docker-compose.yml
   ```

3. **Video Stream Issues**
   ```bash
   # Test RTSP connection
   ffplay rtsp://camera-ip:554/stream
   ```

4. **Service Communication Errors**
   ```bash
   # Check network connectivity
   docker network inspect artemis_default
   ```

### Performance Optimization

1. **Video Processing**
   - Use hardware-accelerated codecs
   - Optimize frame resolution and FPS
   - Implement adaptive bitrate streaming

2. **AI Inference**
   - Use TensorRT for NVIDIA GPUs
   - Implement model quantization
   - Batch processing for efficiency

3. **Database Performance**
   - Configure appropriate indexes
   - Use connection pooling
   - Implement caching strategies

## Security Considerations

### Network Security
```bash
# Configure firewall rules
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API Gateway
sudo ufw enable
```

### SSL/TLS Configuration
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

### Access Control
- Implement role-based access control (RBAC)
- Use strong JWT secrets
- Enable audit logging
- Regular security updates

## Production Considerations

1. **High Availability**
   - Deploy across multiple nodes
   - Use load balancers
   - Implement database replication

2. **Disaster Recovery**
   - Regular backups
   - Offsite storage
   - Recovery procedures testing

3. **Compliance**
   - GDPR compliance for EU deployments
   - SOC 2 Type II for enterprise
   - Regular penetration testing

For support and additional documentation, visit: https://artemis-ai-docs.com
