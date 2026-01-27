// ARTEMIS AI Surveillance Dashboard JavaScript

class DashboardApp {
    constructor() {
        this.data = {
            cameras: [
                {"id": "CAM-001", "name": "Main Entrance", "location": "Building A - Lobby", "status": "online", "x": 100, "y": 150},
                {"id": "CAM-002", "name": "Parking Lot East", "location": "Outdoor - East Side", "status": "online", "x": 300, "y": 100},
                {"id": "CAM-003", "name": "Cafeteria", "location": "Building B - Floor 1", "status": "maintenance", "x": 200, "y": 200},
                {"id": "CAM-004", "name": "Server Room", "location": "Building A - Basement", "status": "online", "x": 150, "y": 180},
                {"id": "CAM-005", "name": "Rooftop Access", "location": "Building C - Roof", "status": "offline", "x": 350, "y": 50},
                {"id": "CAM-006", "name": "Emergency Exit", "location": "Building A - Floor 2", "status": "online", "x": 80, "y": 120},
                {"id": "CAM-007", "name": "Loading Dock", "location": "Building B - Rear", "status": "online", "x": 400, "y": 250},
                {"id": "CAM-008", "name": "Conference Room", "location": "Building A - Floor 3", "status": "online", "x": 120, "y": 90},
                {"id": "CAM-009", "name": "Gym Area", "location": "Building C - Floor 1", "status": "online", "x": 380, "y": 180},
                {"id": "CAM-010", "name": "Storage Area", "location": "Building B - Floor 2", "status": "online", "x": 250, "y": 160},
                {"id": "CAM-011", "name": "Reception", "location": "Building A - Lobby", "status": "online", "x": 90, "y": 140},
                {"id": "CAM-012", "name": "Stairwell A", "location": "Building A - All Floors", "status": "online", "x": 110, "y": 170}
            ],
            threats: [
                {"id": "T001", "type": "Weapon Detection", "severity": "Critical", "camera": "CAM-001", "location": "Main Entrance", "timestamp": "2025-09-18T09:15:32Z", "confidence": 94, "status": "Active"},
                {"id": "T002", "type": "Suspicious Behavior", "severity": "Medium", "camera": "CAM-007", "location": "Loading Dock", "timestamp": "2025-09-18T09:12:18Z", "confidence": 78, "status": "Under Review"},
                {"id": "T003", "type": "Unauthorized Access", "severity": "High", "camera": "CAM-004", "location": "Server Room", "timestamp": "2025-09-18T09:08:45Z", "confidence": 89, "status": "Resolved"},
                {"id": "T004", "type": "Violence", "severity": "Critical", "camera": "CAM-009", "location": "Gym Area", "timestamp": "2025-09-18T09:05:22Z", "confidence": 92, "status": "Active"},
                {"id": "T005", "type": "Vandalism", "severity": "Low", "camera": "CAM-002", "location": "Parking Lot East", "timestamp": "2025-09-18T09:01:33Z", "confidence": 65, "status": "Dismissed"}
            ],
            systemMetrics: {
                cpuUsage: 67,
                memoryUsage: 74,
                networkTraffic: 1250,
                storageUsed: 45,
                activeCameras: 10,
                totalCameras: 12,
                aiInferences: 15420,
                averageLatency: 47
            },
            services: [
                {"name": "Video Ingestion", "status": "healthy", "uptime": 99.8, "language": "C++"},
                {"name": "Computer Vision", "status": "healthy", "uptime": 99.9, "language": "C++/Python"},
                {"name": "AI Inference", "status": "warning", "uptime": 97.2, "language": "Python"},
                {"name": "Threat Intelligence", "status": "healthy", "uptime": 99.5, "language": "Java"},
                {"name": "Event Processing", "status": "healthy", "uptime": 99.7, "language": "Node.js"},
                {"name": "Vector Search", "status": "healthy", "uptime": 98.9, "language": "Python"},
                {"name": "API Gateway", "status": "healthy", "uptime": 99.9, "language": "Node.js"},
                {"name": "Configuration", "status": "healthy", "uptime": 99.3, "language": "Java"}
            ],
            users: [
                {"id": 1, "name": "Sarah Chen", "role": "Security Administrator", "status": "online", "lastActive": "2025-09-18T09:16:45Z"},
                {"id": 2, "name": "Mike Rodriguez", "role": "Operations Manager", "status": "online", "lastActive": "2025-09-18T09:14:32Z"},
                {"id": 3, "name": "Emma Thompson", "role": "Threat Analyst", "status": "offline", "lastActive": "2025-09-18T08:45:21Z"},
                {"id": 4, "name": "David Park", "role": "System Engineer", "status": "online", "lastActive": "2025-09-18T09:16:12Z"}
            ]
        };

        this.charts = {};
        this.selectedCamera = null;
        this.updateInterval = null;
        this.threatTypes = ["Weapon Detection", "Suspicious Behavior", "Unauthorized Access", "Violence", "Vandalism", "Loitering", "Trespassing"];
        this.currentSection = 'live';
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.updateClock();
        this.renderVideoGrid();
        this.renderThreats();
        this.renderMetrics();
        this.renderServices();
        this.renderUsers();
        this.populateCameraSelect();
        this.renderFacilityMap();
        this.startRealTimeUpdates();

        // Set initial section after a small delay to ensure DOM is ready
        setTimeout(() => {
            this.showSection('live');
        }, 100);
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.target.dataset.section;
                console.log('Navigation clicked:', section); // Debug log
                this.showSection(section);
            });
        });

        // Grid layout change
        const gridLayoutSelect = document.getElementById('gridLayout');
        if (gridLayoutSelect) {
            gridLayoutSelect.addEventListener('change', (e) => {
                this.changeGridLayout(e.target.value);
            });
        }

        // PTZ controls
        document.querySelectorAll('.ptz-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handlePTZControl(e.target.dataset.direction);
            });
        });

        // Settings sliders
        const sensitivitySlider = document.getElementById('sensitivity');
        const thresholdSlider = document.getElementById('threshold');
        
        if (sensitivitySlider) {
            sensitivitySlider.addEventListener('input', (e) => {
                const valueSpan = document.getElementById('sensitivityValue');
                if (valueSpan) valueSpan.textContent = e.target.value;
            });
        }
        
        if (thresholdSlider) {
            thresholdSlider.addEventListener('input', (e) => {
                const valueSpan = document.getElementById('thresholdValue');
                if (valueSpan) valueSpan.textContent = e.target.value + '%';
            });
        }

        // Modal
        const closeModalBtn = document.getElementById('closeModal');
        const threatModal = document.getElementById('threatModal');
        
        if (closeModalBtn) {
            closeModalBtn.addEventListener('click', () => {
                this.closeModal();
            });
        }

        if (threatModal) {
            // Close modal on outside click
            threatModal.addEventListener('click', (e) => {
                if (e.target.id === 'threatModal') {
                    this.closeModal();
                }
            });
        }

        // Camera selection
        const cameraSelect = document.getElementById('selectedCamera');
        if (cameraSelect) {
            cameraSelect.addEventListener('change', (e) => {
                this.selectedCamera = e.target.value;
            });
        }

        // Setup threat filters with delay to ensure elements exist
        setTimeout(() => {
            this.setupThreatFilters();
        }, 200);
    }

    setupThreatFilters() {
        const threatSearch = document.getElementById('threatSearch');
        const severityFilter = document.getElementById('severityFilter');
        const statusFilter = document.getElementById('statusFilter');

        if (threatSearch) {
            threatSearch.addEventListener('input', () => this.filterThreats());
        }
        if (severityFilter) {
            severityFilter.addEventListener('change', () => this.filterThreats());
        }
        if (statusFilter) {
            statusFilter.addEventListener('change', () => this.filterThreats());
        }
    }

    showSection(sectionName) {
        console.log('Showing section:', sectionName); // Debug log
        
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        const activeNavBtn = document.querySelector(`[data-section="${sectionName}"]`);
        if (activeNavBtn) {
            activeNavBtn.classList.add('active');
        }

        // Show section
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });
        
        const targetSection = document.getElementById(`${sectionName}-section`);
        if (targetSection) {
            targetSection.classList.add('active');
            this.currentSection = sectionName;
            
            // Initialize charts for analytics section
            if (sectionName === 'analytics') {
                setTimeout(() => this.initCharts(), 100);
            }
            
            // Refresh threats table when showing threats section
            if (sectionName === 'threats') {
                setTimeout(() => this.renderThreatTable(), 100);
            }
        } else {
            console.error('Section not found:', `${sectionName}-section`);
        }
    }

    updateClock() {
        const updateTime = () => {
            const now = new Date();
            const timeString = now.toLocaleString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                timeZoneName: 'short'
            });
            const timeElement = document.getElementById('currentTime');
            if (timeElement) {
                timeElement.textContent = timeString;
            }
        };
        
        updateTime();
        setInterval(updateTime, 1000);
    }

    renderVideoGrid() {
        const grid = document.getElementById('videoGrid');
        if (!grid) return;
        
        grid.className = 'video-grid grid-3x3';
        
        grid.innerHTML = this.data.cameras.map(camera => `
            <div class="video-feed" data-camera="${camera.id}" onclick="app.selectVideoFeed('${camera.id}')">
                <div class="video-content">📹</div>
                <div class="video-overlay">
                    <div class="camera-id">${camera.id}</div>
                    <div class="camera-location">${camera.location}</div>
                </div>
                <div class="video-status">
                    <span class="status-dot status-${camera.status}"></span>
                    <span>${camera.status.toUpperCase()}</span>
                </div>
            </div>
        `).join('');
    }

    selectVideoFeed(cameraId) {
        const camera = this.data.cameras.find(c => c.id === cameraId);
        if (camera) {
            const cameraSelect = document.getElementById('selectedCamera');
            if (cameraSelect) {
                cameraSelect.value = cameraId;
            }
            this.selectedCamera = cameraId;
            
            // Highlight selected feed
            document.querySelectorAll('.video-feed').forEach(feed => {
                feed.classList.remove('selected');
            });
            const selectedFeed = document.querySelector(`[data-camera="${cameraId}"]`);
            if (selectedFeed) {
                selectedFeed.classList.add('selected');
            }
            
            // Show detailed information
            this.showCameraDetails(camera);
        }
    }

    showCameraDetails(camera) {
        // Find related threats
        const relatedThreats = this.data.threats.filter(t => t.camera === camera.id);
        
        const modalBody = document.getElementById('threatModalBody');
        if (!modalBody) return;

        modalBody.innerHTML = `
            <div class="camera-detail">
                <div class="detail-row">
                    <strong>Camera ID:</strong> ${camera.id}
                </div>
                <div class="detail-row">
                    <strong>Name:</strong> ${camera.name}
                </div>
                <div class="detail-row">
                    <strong>Location:</strong> ${camera.location}
                </div>
                <div class="detail-row">
                    <strong>Status:</strong> 
                    <span class="status-badge status-${camera.status}">${camera.status.toUpperCase()}</span>
                </div>
                ${relatedThreats.length > 0 ? `
                <div class="detail-row">
                    <strong>Recent Threats:</strong>
                    <div style="margin-top: 8px;">
                        ${relatedThreats.map(threat => `
                            <div style="margin-bottom: 4px;">
                                <span class="severity-badge severity-${threat.severity.toLowerCase()}">${threat.severity}</span>
                                ${threat.type} - ${this.formatTime(threat.timestamp)}
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : '<div class="detail-row"><strong>No recent threats detected</strong></div>'}
                <div class="detail-actions" style="margin-top: 20px;">
                    <button class="btn btn--primary" onclick="app.closeModal()">Close</button>
                </div>
            </div>
        `;
        
        const modal = document.getElementById('threatModal');
        if (modal) {
            modal.classList.remove('hidden');
        }
    }

    changeGridLayout(layout) {
        const grid = document.getElementById('videoGrid');
        if (grid) {
            grid.className = `video-grid grid-${layout}`;
        }
    }

    renderThreats() {
        // Sidebar threats
        const threatList = document.getElementById('threatList');
        if (threatList) {
            const activeThreats = this.data.threats.filter(t => t.status === 'Active');
            
            threatList.innerHTML = activeThreats.map(threat => `
                <div class="threat-item" onclick="app.showThreatDetails('${threat.id}')">
                    <div class="threat-header">
                        <span class="threat-type">${threat.type}</span>
                        <span class="severity-badge severity-${threat.severity.toLowerCase()}">${threat.severity}</span>
                    </div>
                    <div class="threat-details">
                        <div>${threat.location}</div>
                        <div>${this.formatTime(threat.timestamp)}</div>
                        <div>Confidence: ${threat.confidence}%</div>
                    </div>
                </div>
            `).join('');
        }

        // Update threat counts
        this.updateThreatCounts();
        
        // Render threat table if we're on threats section
        if (this.currentSection === 'threats') {
            this.renderThreatTable();
        }
    }

    updateThreatCounts() {
        const counts = {
            Critical: 0,
            High: 0,
            Medium: 0,
            Low: 0
        };

        this.data.threats.forEach(threat => {
            if (threat.status === 'Active') {
                counts[threat.severity]++;
            }
        });

        const criticalCount = document.getElementById('criticalCount');
        const highCount = document.getElementById('highCount');
        const mediumCount = document.getElementById('mediumCount');
        const lowCount = document.getElementById('lowCount');

        if (criticalCount) criticalCount.textContent = counts.Critical;
        if (highCount) highCount.textContent = counts.High;
        if (mediumCount) mediumCount.textContent = counts.Medium;
        if (lowCount) lowCount.textContent = counts.Low;
    }

    renderThreatTable() {
        const tbody = document.getElementById('threatTableBody');
        if (!tbody) return;
        
        const threats = this.getFilteredThreats();
        
        tbody.innerHTML = threats.map(threat => `
            <tr onclick="app.showThreatDetails('${threat.id}')" style="cursor: pointer;">
                <td>${threat.id}</td>
                <td>${threat.type}</td>
                <td><span class="severity-badge severity-${threat.severity.toLowerCase()}">${threat.severity}</span></td>
                <td>${threat.location}</td>
                <td>${this.formatTime(threat.timestamp)}</td>
                <td>${threat.confidence}%</td>
                <td><span class="status-${threat.status.toLowerCase().replace(' ', '-')}">${threat.status}</span></td>
                <td><button class="btn btn--sm btn--secondary" onclick="event.stopPropagation(); app.investigateThreat('${threat.id}')">Investigate</button></td>
            </tr>
        `).join('');
    }

    getFilteredThreats() {
        const searchElement = document.getElementById('threatSearch');
        const severityFilterElement = document.getElementById('severityFilter');
        const statusFilterElement = document.getElementById('statusFilter');
        
        const search = searchElement ? searchElement.value.toLowerCase() : '';
        const severityFilter = severityFilterElement ? severityFilterElement.value : '';
        const statusFilter = statusFilterElement ? statusFilterElement.value : '';

        return this.data.threats.filter(threat => {
            const matchesSearch = !search || 
                threat.type.toLowerCase().includes(search) ||
                threat.location.toLowerCase().includes(search) ||
                threat.id.toLowerCase().includes(search);
            
            const matchesSeverity = !severityFilter || threat.severity === severityFilter;
            const matchesStatus = !statusFilter || threat.status === statusFilter;
            
            return matchesSearch && matchesSeverity && matchesStatus;
        });
    }

    filterThreats() {
        this.renderThreatTable();
    }

    investigateThreat(threatId) {
        this.showThreatDetails(threatId);
    }

    showThreatDetails(threatId) {
        const threat = this.data.threats.find(t => t.id === threatId);
        if (!threat) return;

        const camera = this.data.cameras.find(c => c.id === threat.camera);
        const modalBody = document.getElementById('threatModalBody');
        if (!modalBody) return;
        
        modalBody.innerHTML = `
            <div class="threat-detail">
                <div class="detail-row">
                    <strong>Threat ID:</strong> ${threat.id}
                </div>
                <div class="detail-row">
                    <strong>Type:</strong> ${threat.type}
                </div>
                <div class="detail-row">
                    <strong>Severity:</strong> 
                    <span class="severity-badge severity-${threat.severity.toLowerCase()}">${threat.severity}</span>
                </div>
                <div class="detail-row">
                    <strong>Location:</strong> ${threat.location}
                </div>
                <div class="detail-row">
                    <strong>Camera:</strong> ${threat.camera} - ${camera ? camera.name : 'Unknown'}
                </div>
                <div class="detail-row">
                    <strong>Timestamp:</strong> ${new Date(threat.timestamp).toLocaleString()}
                </div>
                <div class="detail-row">
                    <strong>Confidence:</strong> ${threat.confidence}%
                </div>
                <div class="detail-row">
                    <strong>Status:</strong> ${threat.status}
                </div>
                <div class="detail-actions" style="margin-top: 20px;">
                    <button class="btn btn--primary" onclick="app.updateThreatStatus('${threat.id}', 'Under Review')">Review</button>
                    <button class="btn btn--secondary" onclick="app.updateThreatStatus('${threat.id}', 'Resolved')">Resolve</button>
                    <button class="btn btn--outline" onclick="app.updateThreatStatus('${threat.id}', 'Dismissed')">Dismiss</button>
                </div>
            </div>
        `;
        
        const modal = document.getElementById('threatModal');
        if (modal) {
            modal.classList.remove('hidden');
        }
    }

    updateThreatStatus(threatId, newStatus) {
        const threat = this.data.threats.find(t => t.id === threatId);
        if (threat) {
            threat.status = newStatus;
            this.renderThreats();
            this.renderFacilityMap();
            this.closeModal();
            this.showNotification(`Threat ${threatId} status updated to ${newStatus}`);
        }
    }

    closeModal() {
        const modal = document.getElementById('threatModal');
        if (modal) {
            modal.classList.add('hidden');
        }
    }

    renderMetrics() {
        const metricsGrid = document.getElementById('metricsGrid');
        if (!metricsGrid) return;
        
        const metrics = this.data.systemMetrics;
        
        metricsGrid.innerHTML = `
            <div class="metric-item">
                <div class="metric-value">${Math.round(metrics.cpuUsage)}%</div>
                <div class="metric-label">CPU</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${Math.round(metrics.memoryUsage)}%</div>
                <div class="metric-label">Memory</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${metrics.activeCameras}</div>
                <div class="metric-label">Active Cams</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${Math.round(metrics.averageLatency)}ms</div>
                <div class="metric-label">Latency</div>
            </div>
        `;
    }

    renderServices() {
        const serviceList = document.getElementById('serviceList');
        if (!serviceList) return;
        
        serviceList.innerHTML = this.data.services.map(service => `
            <div class="service-item">
                <div>
                    <div style="font-weight: 500;">${service.name}</div>
                    <div style="font-size: 12px; color: var(--color-text-secondary);">${service.language} • ${service.uptime}% uptime</div>
                </div>
                <span class="service-status status-${service.status}">${service.status}</span>
            </div>
        `).join('');
    }

    renderUsers() {
        const userList = document.getElementById('userList');
        if (!userList) return;
        
        userList.innerHTML = this.data.users.map(user => `
            <div class="user-item">
                <div>
                    <div style="font-weight: 500;">${user.name}</div>
                    <div style="font-size: 12px; color: var(--color-text-secondary);">${user.role}</div>
                </div>
                <span class="user-status status-${user.status}">${user.status}</span>
            </div>
        `).join('');
    }

    populateCameraSelect() {
        const select = document.getElementById('selectedCamera');
        if (!select) return;
        
        select.innerHTML = '<option value="">Select Camera</option>' + 
            this.data.cameras.map(camera => 
                `<option value="${camera.id}">${camera.id} - ${camera.name}</option>`
            ).join('');
    }

    handlePTZControl(direction) {
        if (!this.selectedCamera) {
            this.showNotification('Please select a camera first');
            return;
        }
        
        console.log(`PTZ Control: ${direction} for camera ${this.selectedCamera}`);
        this.showNotification(`PTZ ${direction} command sent to ${this.selectedCamera}`);
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: var(--color-primary);
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            z-index: 1001;
            font-size: 14px;
            max-width: 300px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    initCharts() {
        // Clear existing charts
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.destroy) {
                chart.destroy();
            }
        });
        
        // Initialize charts
        this.initThreatTrendsChart();
        this.initPerformanceChart();
        this.initAccuracyChart();
    }

    initThreatTrendsChart() {
        const ctx = document.getElementById('threatTrendsChart');
        if (!ctx) return;

        const hours = Array.from({length: 24}, (_, i) => `${i}:00`);
        const data = hours.map(() => Math.floor(Math.random() * 10));

        this.charts.threatTrends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: hours,
                datasets: [{
                    label: 'Threats Detected',
                    data: data,
                    borderColor: '#1FB8CD',
                    backgroundColor: 'rgba(31, 184, 205, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f5f5f5' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#f5f5f5' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { color: '#f5f5f5' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    initPerformanceChart() {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;

        this.charts.performance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['CPU', 'Memory', 'Storage', 'Network'],
                datasets: [{
                    data: [67, 74, 45, 82],
                    backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#5D878F']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f5f5f5' }
                    }
                }
            }
        });
    }

    initAccuracyChart() {
        const ctx = document.getElementById('accuracyChart');
        if (!ctx) return;

        this.charts.accuracy = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Weapon Detection', 'Suspicious Behavior', 'Unauthorized Access', 'Violence', 'Vandalism'],
                datasets: [{
                    label: 'Accuracy %',
                    data: [94, 78, 89, 92, 65],
                    backgroundColor: '#1FB8CD'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f5f5f5' }
                    }
                },
                scales: {
                    x: {
                        ticks: { 
                            color: '#f5f5f5',
                            maxRotation: 45
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { color: '#f5f5f5' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    renderFacilityMap() {
        const map = document.getElementById('facilityMap');
        if (!map) return;
        
        map.innerHTML = `
            <div style="position: relative; width: 100%; height: 100%; background: linear-gradient(45deg, #1a1f2e 0%, #2a3f5e 100%); border-radius: 8px;">
                ${this.data.cameras.map(camera => `
                    <div class="map-point camera" 
                         style="left: ${camera.x}px; top: ${camera.y}px;" 
                         title="${camera.name} - ${camera.location}">
                    </div>
                `).join('')}
                ${this.data.threats.filter(t => t.status === 'Active').map(threat => {
                    const camera = this.data.cameras.find(c => c.id === threat.camera);
                    return camera ? `
                        <div class="map-point threat" 
                             style="left: ${camera.x + 15}px; top: ${camera.y + 15}px;" 
                             title="${threat.type} - ${threat.location}">
                        </div>
                    ` : '';
                }).join('')}
            </div>
        `;
    }

    startRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        // Simulate real-time updates every 2 seconds
        this.updateInterval = setInterval(() => {
            this.simulateRealTimeData();
        }, 2000);
    }

    simulateRealTimeData() {
        // Simulate new threats occasionally
        if (Math.random() < 0.1) { // 10% chance every 2 seconds
            this.generateRandomThreat();
        }

        // Update system metrics
        this.updateSystemMetrics();
        
        // Update service status occasionally
        if (Math.random() < 0.05) { // 5% chance every 2 seconds
            this.updateServiceStatus();
        }
    }

    generateRandomThreat() {
        const threatId = `T${String(this.data.threats.length + 1).padStart(3, '0')}`;
        const threatType = this.threatTypes[Math.floor(Math.random() * this.threatTypes.length)];
        const severity = ['Low', 'Medium', 'High', 'Critical'][Math.floor(Math.random() * 4)];
        const camera = this.data.cameras[Math.floor(Math.random() * this.data.cameras.length)];
        const confidence = Math.floor(Math.random() * 40) + 60; // 60-99%
        
        const newThreat = {
            id: threatId,
            type: threatType,
            severity: severity,
            camera: camera.id,
            location: camera.location,
            timestamp: new Date().toISOString(),
            confidence: confidence,
            status: 'Active'
        };
        
        this.data.threats.unshift(newThreat);
        
        // Keep only last 20 threats
        if (this.data.threats.length > 20) {
            this.data.threats.pop();
        }
        
        this.renderThreats();
        this.renderFacilityMap();
        this.showNotification(`New ${severity.toLowerCase()} threat detected: ${threatType}`);
    }

    updateSystemMetrics() {
        const metrics = this.data.systemMetrics;
        
        // Simulate realistic variations
        metrics.cpuUsage += (Math.random() - 0.5) * 10;
        metrics.cpuUsage = Math.max(0, Math.min(100, metrics.cpuUsage));
        
        metrics.memoryUsage += (Math.random() - 0.5) * 5;
        metrics.memoryUsage = Math.max(0, Math.min(100, metrics.memoryUsage));
        
        metrics.averageLatency += (Math.random() - 0.5) * 20;
        metrics.averageLatency = Math.max(10, Math.min(200, metrics.averageLatency));
        
        metrics.aiInferences += Math.floor(Math.random() * 100);
        
        this.renderMetrics();
    }

    updateServiceStatus() {
        const service = this.data.services[Math.floor(Math.random() * this.data.services.length)];
        const statuses = ['healthy', 'warning', 'error'];
        
        // Mostly stay healthy, occasionally change
        if (Math.random() < 0.8) {
            service.status = 'healthy';
        } else {
            service.status = statuses[Math.floor(Math.random() * statuses.length)];
        }
        
        this.renderServices();
    }

    formatTime(timestamp) {
        return new Date(timestamp).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    // Cleanup method
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.destroy) {
                chart.destroy();
            }
        });
    }
}

// Initialize the application
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new DashboardApp();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && app) {
        app.startRealTimeUpdates();
    } else if (app && app.updateInterval) {
        clearInterval(app.updateInterval);
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (!app) return;
    
    if (e.altKey) {
        switch(e.key) {
            case '1':
                e.preventDefault();
                app.showSection('live');
                break;
            case '2':
                e.preventDefault();
                app.showSection('threats');
                break;
            case '3':
                e.preventDefault();
                app.showSection('analytics');
                break;
            case '4':
                e.preventDefault();
                app.showSection('settings');
                break;
        }
    }
    
    if (e.key === 'Escape') {
        app.closeModal();
    }
});