// server.js - Node.js API Gateway with Express
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const jwt = require('jsonwebtoken');
const redis = require('redis');
const { createProxyMiddleware } = require('http-proxy-middleware');
const WebSocket = require('ws');
const http = require('http');

const app = express();
const server = http.createServer(app);

// Configuration
const config = {
    port: process.env.PORT || 8000,
    jwtSecret: process.env.JWT_SECRET || 'your-super-secret-jwt-key',
    redisUrl: process.env.REDIS_URL || 'redis://redis:6379',
    rateLimitRequests: parseInt(process.env.RATE_LIMIT_REQUESTS) || 1000,
    rateLimitWindow: parseInt(process.env.RATE_LIMIT_WINDOW) || 60000
};

// Redis client for session storage and caching
const redisClient = redis.createClient({
    url: config.redisUrl,
    retry_strategy: (options) => {
        if (options.error && options.error.code === 'ECONNREFUSED') {
            return new Error('Redis server refused connection');
        }
        if (options.total_retry_time > 1000 * 60 * 60) {
            return new Error('Retry time exhausted');
        }
        if (options.attempt > 10) {
            return undefined;
        }
        return Math.min(options.attempt * 100, 3000);
    }
});

redisClient.connect();

// Middleware setup
app.use(helmet());
app.use(cors({
    origin: ['http://localhost:3000', 'https://artemis-dashboard.com'],
    credentials: true
}));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate limiting
const limiter = rateLimit({
    windowMs: config.rateLimitWindow,
    max: config.rateLimitRequests,
    message: {
        error: 'Too many requests from this IP, please try again later.',
        retryAfter: Math.ceil(config.rateLimitWindow / 1000)
    },
    standardHeaders: true,
    legacyHeaders: false,
});

app.use(limiter);

// JWT Authentication middleware
const authenticateToken = async (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ error: 'Access token required' });
    }

    try {
        const decoded = jwt.verify(token, config.jwtSecret);

        // Check if token is blacklisted in Redis
        const isBlacklisted = await redisClient.get(`blacklist:${token}`);
        if (isBlacklisted) {
            return res.status(401).json({ error: 'Token has been revoked' });
        }

        req.user = decoded;
        next();
    } catch (error) {
        return res.status(403).json({ error: 'Invalid or expired token' });
    }
};

// Role-based access control
const requireRole = (roles) => {
    return (req, res, next) => {
        if (!req.user) {
            return res.status(401).json({ error: 'Authentication required' });
        }

        if (!roles.includes(req.user.role)) {
            return res.status(403).json({ error: 'Insufficient permissions' });
        }

        next();
    };
};

// Service discovery and health checking
const services = {
    'video-ingestion': {
        url: 'http://video-ingestion:8001',
        health: '/health',
        status: 'unknown'
    },
    'computer-vision': {
        url: 'http://computer-vision:8002',
        health: '/health', 
        status: 'unknown'
    },
    'ai-inference': {
        url: 'http://ai-inference:8003',
        health: '/health',
        status: 'unknown'
    },
    'threat-intelligence': {
        url: 'http://threat-intelligence:8004',
        health: '/health',
        status: 'unknown'
    },
    'configuration': {
        url: 'http://configuration-service:8005',
        health: '/health',
        status: 'unknown'
    },
    'event-processing': {
        url: 'http://event-processing:8006',
        health: '/health',
        status: 'unknown'
    },
    'vector-search': {
        url: 'http://vector-search:8007',
        health: '/health',
        status: 'unknown'
    }
};

// Health check for downstream services
const checkServiceHealth = async (serviceName, serviceConfig) => {
    try {
        const response = await fetch(`${serviceConfig.url}${serviceConfig.health}`, {
            method: 'GET',
            timeout: 5000
        });

        if (response.ok) {
            services[serviceName].status = 'healthy';
            return true;
        } else {
            services[serviceName].status = 'unhealthy';
            return false;
        }
    } catch (error) {
        services[serviceName].status = 'error';
        console.error(`Health check failed for ${serviceName}:`, error.message);
        return false;
    }
};

// Periodic health checks
setInterval(async () => {
    for (const [serviceName, serviceConfig] of Object.entries(services)) {
        await checkServiceHealth(serviceName, serviceConfig);
    }
}, 30000); // Check every 30 seconds

// Load balancing for multiple instances
const createLoadBalancer = (serviceInstances) => {
    let currentIndex = 0;

    return (req, res, next) => {
        const healthyInstances = serviceInstances.filter(instance => 
            services[instance] && services[instance].status === 'healthy'
        );

        if (healthyInstances.length === 0) {
            return res.status(503).json({ error: 'Service temporarily unavailable' });
        }

        const selectedInstance = healthyInstances[currentIndex % healthyInstances.length];
        currentIndex++;

        req.selectedService = services[selectedInstance];
        next();
    };
};

// Proxy middleware with circuit breaker pattern
const createServiceProxy = (target, pathRewrite = {}) => {
    return createProxyMiddleware({
        target,
        changeOrigin: true,
        pathRewrite,
        timeout: 30000,
        onError: (err, req, res) => {
            console.error(`Proxy error for ${target}:`, err.message);
            res.status(502).json({
                error: 'Bad Gateway',
                message: 'Service temporarily unavailable',
                timestamp: new Date().toISOString()
            });
        },
        onProxyReq: (proxyReq, req, res) => {
            // Add request tracking
            const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            proxyReq.setHeader('X-Request-ID', requestId);
            proxyReq.setHeader('X-Forwarded-For', req.ip);

            if (req.user) {
                proxyReq.setHeader('X-User-ID', req.user.id);
                proxyReq.setHeader('X-User-Role', req.user.role);
            }
        },
        onProxyRes: (proxyRes, req, res) => {
            // Add CORS headers to proxied responses
            proxyRes.headers['Access-Control-Allow-Origin'] = '*';
            proxyRes.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS';
            proxyRes.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization';
        }
    });
};

// API Routes with authentication and authorization

// Authentication endpoints
app.post('/auth/login', async (req, res) => {
    const { username, password } = req.body;

    // In production, validate against user database
    if (username === 'admin' && password === 'artemis123') {
        const token = jwt.sign(
            { 
                id: 1, 
                username: 'admin', 
                role: 'administrator',
                permissions: ['read', 'write', 'admin']
            },
            config.jwtSecret,
            { expiresIn: '24h' }
        );

        // Store session in Redis
        await redisClient.setEx(`session:${token}`, 86400, JSON.stringify({
            userId: 1,
            username: 'admin',
            loginTime: new Date().toISOString()
        }));

        res.json({ 
            token, 
            user: { 
                id: 1, 
                username: 'admin', 
                role: 'administrator' 
            },
            expiresIn: '24h'
        });
    } else {
        res.status(401).json({ error: 'Invalid credentials' });
    }
});

app.post('/auth/logout', authenticateToken, async (req, res) => {
    const token = req.headers['authorization'].split(' ')[1];

    // Add token to blacklist
    await redisClient.setEx(`blacklist:${token}`, 86400, 'revoked');

    res.json({ message: 'Successfully logged out' });
});

// Service proxy routes with authentication
app.use('/api/v1/video', 
    authenticateToken, 
    requireRole(['administrator', 'operator']),
    createServiceProxy('http://video-ingestion:8001', { '^/api/v1/video': '' })
);

app.use('/api/v1/vision', 
    authenticateToken,
    requireRole(['administrator', 'operator', 'analyst']),
    createServiceProxy('http://computer-vision:8002', { '^/api/v1/vision': '' })
);

app.use('/api/v1/inference', 
    authenticateToken,
    requireRole(['administrator', 'operator', 'analyst']),
    createServiceProxy('http://ai-inference:8003', { '^/api/v1/inference': '' })
);

app.use('/api/v1/threats', 
    authenticateToken,
    requireRole(['administrator', 'operator', 'analyst']),
    createServiceProxy('http://threat-intelligence:8004', { '^/api/v1/threats': '' })
);

app.use('/api/v1/config', 
    authenticateToken,
    requireRole(['administrator']),
    createServiceProxy('http://configuration-service:8005', { '^/api/v1/config': '' })
);

app.use('/api/v1/search', 
    authenticateToken,
    requireRole(['administrator', 'operator', 'analyst']),
    createServiceProxy('http://vector-search:8007', { '^/api/v1/search': '' })
);

// System monitoring and health endpoints
app.get('/api/system/health', authenticateToken, async (req, res) => {
    const systemHealth = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        services: services,
        gateway: {
            uptime: process.uptime(),
            memory: process.memoryUsage(),
            version: process.version
        }
    };

    res.json(systemHealth);
});

app.get('/api/system/metrics', authenticateToken, requireRole(['administrator']), (req, res) => {
    const metrics = {
        requests_per_minute: Math.floor(Math.random() * 1000),
        active_connections: Math.floor(Math.random() * 100),
        error_rate: Math.random() * 0.05,
        avg_response_time: Math.random() * 200,
        cpu_usage: Math.random() * 100,
        memory_usage: (process.memoryUsage().heapUsed / process.memoryUsage().heapTotal) * 100
    };

    res.json(metrics);
});

// WebSocket proxy for real-time communication
const wss = new WebSocket.Server({ server, path: '/ws' });

wss.on('connection', (ws, req) => {
    console.log('New WebSocket connection established');

    // Authenticate WebSocket connection
    const token = new URL(req.url, `http://${req.headers.host}`).searchParams.get('token');

    if (!token) {
        ws.close(1008, 'Authentication required');
        return;
    }

    try {
        const decoded = jwt.verify(token, config.jwtSecret);
        ws.user = decoded;

        // Forward to event processing service
        const eventServiceWs = new WebSocket('ws://event-processing:8006/ws');

        // Proxy messages between client and event service
        ws.on('message', (message) => {
            if (eventServiceWs.readyState === WebSocket.OPEN) {
                eventServiceWs.send(message);
            }
        });

        eventServiceWs.on('message', (message) => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(message);
            }
        });

        // Handle connection cleanup
        ws.on('close', () => {
            eventServiceWs.close();
        });

        eventServiceWs.on('close', () => {
            ws.close();
        });

    } catch (error) {
        ws.close(1008, 'Invalid token');
    }
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Gateway error:', error);

    res.status(error.status || 500).json({
        error: 'Internal Server Error',
        message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong',
        timestamp: new Date().toISOString(),
        requestId: req.headers['x-request-id']
    });
});

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({
        error: 'Not Found',
        message: `Route ${req.originalUrl} not found`,
        timestamp: new Date().toISOString()
    });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    console.log('Received SIGTERM, shutting down gracefully');

    // Close WebSocket server
    wss.close();

    // Close Redis connection
    await redisClient.quit();

    // Close HTTP server
    server.close(() => {
        console.log('Gateway server closed');
        process.exit(0);
    });
});

// Start the server
server.listen(config.port, '0.0.0.0', () => {
    console.log(`ARTEMIS API Gateway listening on port ${config.port}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log('Services configured:', Object.keys(services).join(', '));
});

module.exports = app;
