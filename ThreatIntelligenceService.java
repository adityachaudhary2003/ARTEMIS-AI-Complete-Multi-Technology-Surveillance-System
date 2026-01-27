// ThreatIntelligenceService.java - Java Spring Boot microservice
package com.artemis.threat;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.stereotype.Repository;
import org.springframework.http.ResponseEntity;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@SpringBootApplication
public class ThreatIntelligenceApplication {
    public static void main(String[] args) {
        SpringApplication.run(ThreatIntelligenceApplication.class, args);
    }
}

// Threat Intelligence Entity
@Entity
@Table(name = "threat_intelligence")
public class ThreatIntelligence {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotNull
    @Column(name = "threat_id")
    private String threatId;

    @NotNull
    @Column(name = "threat_type")
    private String threatType;

    @NotNull
    @Column(name = "severity")
    @Enumerated(EnumType.STRING)
    private ThreatSeverity severity;

    @Column(name = "description", length = 1000)
    private String description;

    @Column(name = "confidence_score")
    private Double confidenceScore;

    @Column(name = "camera_id")
    private String cameraId;

    @Column(name = "location")
    private String location;

    @Column(name = "detected_at")
    private LocalDateTime detectedAt;

    @Column(name = "status")
    @Enumerated(EnumType.STRING)
    private ThreatStatus status;

    @ElementCollection
    @CollectionTable(name = "threat_indicators")
    private List<String> indicators;

    // Constructors, getters, and setters
    public ThreatIntelligence() {}

    public ThreatIntelligence(String threatId, String threatType, ThreatSeverity severity) {
        this.threatId = threatId;
        this.threatType = threatType;
        this.severity = severity;
        this.detectedAt = LocalDateTime.now();
        this.status = ThreatStatus.ACTIVE;
        this.indicators = new ArrayList<>();
    }

    // Getters and setters omitted for brevity
    // ... getter and setter methods
}

// Enums
enum ThreatSeverity {
    LOW, MEDIUM, HIGH, CRITICAL
}

enum ThreatStatus {
    ACTIVE, UNDER_REVIEW, RESOLVED, DISMISSED
}

// Repository Interface
@Repository
public interface ThreatIntelligenceRepository extends JpaRepository<ThreatIntelligence, Long> {
    List<ThreatIntelligence> findByThreatTypeAndStatus(String threatType, ThreatStatus status);
    List<ThreatIntelligence> findByCameraIdAndDetectedAtAfter(String cameraId, LocalDateTime after);
    List<ThreatIntelligence> findBySeverityAndStatus(ThreatSeverity severity, ThreatStatus status);
    List<ThreatIntelligence> findByStatusOrderByDetectedAtDesc(ThreatStatus status);
}

// Threat Analysis Service
@Service
public class ThreatAnalysisService {

    @Autowired
    private ThreatIntelligenceRepository repository;

    @Autowired
    private KafkaTemplate<String, Object> kafkaTemplate;

    private final Map<String, List<ThreatIntelligence>> patternCache = new ConcurrentHashMap<>();

    // Threat pattern matching weights
    private final Map<String, Double> threatWeights = Map.of(
        "weapon_detection", 1.0,
        "violence", 0.95,
        "unauthorized_access", 0.85,
        "suspicious_behavior", 0.6,
        "vandalism", 0.4
    );

    public ThreatIntelligence analyzeThreat(ThreatDetectionRequest request) {
        // Create new threat intelligence record
        ThreatIntelligence threat = new ThreatIntelligence(
            generateThreatId(),
            request.getThreatType(),
            calculateSeverity(request)
        );

        threat.setCameraId(request.getCameraId());
        threat.setLocation(request.getLocation());
        threat.setConfidenceScore(request.getConfidenceScore());
        threat.setDescription(generateThreatDescription(request));
        threat.setIndicators(extractThreatIndicators(request));

        // Correlate with historical patterns
        correlateWithHistoricalThreats(threat);

        // Save to database
        threat = repository.save(threat);

        // Publish threat alert to Kafka
        publishThreatAlert(threat);

        // Update pattern cache
        updatePatternCache(threat);

        return threat;
    }

    public List<ThreatIntelligence> getActiveThreatsByCamera(String cameraId, int hours) {
        LocalDateTime since = LocalDateTime.now().minusHours(hours);
        return repository.findByCameraIdAndDetectedAtAfter(cameraId, since)
                        .stream()
                        .filter(t -> t.getStatus() == ThreatStatus.ACTIVE)
                        .collect(Collectors.toList());
    }

    public ThreatRiskAssessment assessLocationRisk(String location) {
        List<ThreatIntelligence> recentThreats = repository.findAll()
            .stream()
            .filter(t -> t.getLocation().equals(location))
            .filter(t -> t.getDetectedAt().isAfter(LocalDateTime.now().minusDays(7)))
            .collect(Collectors.toList());

        double riskScore = calculateLocationRiskScore(recentThreats);
        return new ThreatRiskAssessment(location, riskScore, recentThreats.size());
    }

    private ThreatSeverity calculateSeverity(ThreatDetectionRequest request) {
        double score = request.getConfidenceScore() * 
                      threatWeights.getOrDefault(request.getThreatType().toLowerCase(), 0.5);

        if (score >= 0.9) return ThreatSeverity.CRITICAL;
        if (score >= 0.7) return ThreatSeverity.HIGH;
        if (score >= 0.5) return ThreatSeverity.MEDIUM;
        return ThreatSeverity.LOW;
    }

    private void correlateWithHistoricalThreats(ThreatIntelligence threat) {
        // Find similar threats in the last 30 days
        LocalDateTime thirtyDaysAgo = LocalDateTime.now().minusDays(30);
        List<ThreatIntelligence> similarThreats = repository.findAll()
            .stream()
            .filter(t -> t.getThreatType().equals(threat.getThreatType()))
            .filter(t -> t.getDetectedAt().isAfter(thirtyDaysAgo))
            .filter(t -> t.getLocation().equals(threat.getLocation()))
            .collect(Collectors.toList());

        if (similarThreats.size() > 3) {
            // Escalate severity due to pattern
            threat.setSeverity(ThreatSeverity.HIGH);
            threat.getIndicators().add("PATTERN_ESCALATION");
        }
    }

    private void publishThreatAlert(ThreatIntelligence threat) {
        ThreatAlertMessage alert = new ThreatAlertMessage(
            threat.getThreatId(),
            threat.getThreatType(),
            threat.getSeverity().toString(),
            threat.getLocation(),
            threat.getConfidenceScore(),
            threat.getDetectedAt()
        );

        kafkaTemplate.send("threat-alerts", alert);
    }

    private String generateThreatId() {
        return "THR-" + System.currentTimeMillis() + "-" + 
               String.format("%04d", new Random().nextInt(10000));
    }

    private String generateThreatDescription(ThreatDetectionRequest request) {
        return String.format("Detected %s with %.1f%% confidence at %s", 
            request.getThreatType(), 
            request.getConfidenceScore() * 100,
            request.getLocation());
    }

    private List<String> extractThreatIndicators(ThreatDetectionRequest request) {
        List<String> indicators = new ArrayList<>();

        if (request.getConfidenceScore() > 0.9) {
            indicators.add("HIGH_CONFIDENCE");
        }
        if (request.getMetadata().containsKey("multiple_objects")) {
            indicators.add("MULTIPLE_OBJECTS");
        }
        if (request.getMetadata().containsKey("movement_pattern")) {
            indicators.add("UNUSUAL_MOVEMENT");
        }

        return indicators;
    }

    private void updatePatternCache(ThreatIntelligence threat) {
        String key = threat.getThreatType() + "_" + threat.getLocation();
        patternCache.computeIfAbsent(key, k -> new ArrayList<>()).add(threat);

        // Keep only last 100 entries per pattern
        List<ThreatIntelligence> patterns = patternCache.get(key);
        if (patterns.size() > 100) {
            patterns.subList(0, patterns.size() - 100).clear();
        }
    }

    private double calculateLocationRiskScore(List<ThreatIntelligence> threats) {
        if (threats.isEmpty()) return 0.0;

        double severityScore = threats.stream()
            .mapToDouble(t -> {
                switch (t.getSeverity()) {
                    case CRITICAL: return 4.0;
                    case HIGH: return 3.0;
                    case MEDIUM: return 2.0;
                    case LOW: return 1.0;
                    default: return 0.5;
                }
            })
            .average()
            .orElse(0.0);

        double frequencyScore = Math.min(threats.size() / 10.0, 2.0);

        return Math.min((severityScore + frequencyScore) / 6.0, 1.0);
    }
}

// REST Controller
@RestController
@RequestMapping("/api/v1/threats")
public class ThreatIntelligenceController {

    @Autowired
    private ThreatAnalysisService analysisService;

    @PostMapping("/analyze")
    public ResponseEntity<ThreatIntelligence> analyzeThreat(
            @RequestBody ThreatDetectionRequest request) {
        try {
            ThreatIntelligence result = analysisService.analyzeThreat(request);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }

    @GetMapping("/camera/{cameraId}")
    public ResponseEntity<List<ThreatIntelligence>> getThreatsByCamera(
            @PathVariable String cameraId,
            @RequestParam(defaultValue = "24") int hours) {
        List<ThreatIntelligence> threats = analysisService.getActiveThreatsByCamera(cameraId, hours);
        return ResponseEntity.ok(threats);
    }

    @GetMapping("/risk-assessment/{location}")
    public ResponseEntity<ThreatRiskAssessment> assessLocationRisk(
            @PathVariable String location) {
        ThreatRiskAssessment assessment = analysisService.assessLocationRisk(location);
        return ResponseEntity.ok(assessment);
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> healthCheck() {
        Map<String, Object> health = new HashMap<>();
        health.put("status", "healthy");
        health.put("service", "Threat Intelligence");
        health.put("timestamp", LocalDateTime.now());
        return ResponseEntity.ok(health);
    }
}

// Kafka Listener for real-time threat detection
@Component
public class ThreatDetectionListener {

    @Autowired
    private ThreatAnalysisService analysisService;

    @KafkaListener(topics = "ai-detections", groupId = "threat-intelligence-group")
    public void handleThreatDetection(ThreatDetectionRequest request) {
        try {
            analysisService.analyzeThreat(request);
        } catch (Exception e) {
            // Log error and continue processing
            System.err.println("Error processing threat detection: " + e.getMessage());
        }
    }
}

// Request/Response DTOs
class ThreatDetectionRequest {
    private String threatType;
    private Double confidenceScore;
    private String cameraId;
    private String location;
    private Map<String, Object> metadata;

    // Constructors, getters, and setters
    public ThreatDetectionRequest() {}

    // Getters and setters omitted for brevity
}

class ThreatAlertMessage {
    private String threatId;
    private String threatType;
    private String severity;
    private String location;
    private Double confidenceScore;
    private LocalDateTime detectedAt;

    public ThreatAlertMessage(String threatId, String threatType, String severity, 
                             String location, Double confidenceScore, LocalDateTime detectedAt) {
        this.threatId = threatId;
        this.threatType = threatType;
        this.severity = severity;
        this.location = location;
        this.confidenceScore = confidenceScore;
        this.detectedAt = detectedAt;
    }

    // Getters and setters omitted for brevity
}

class ThreatRiskAssessment {
    private String location;
    private Double riskScore;
    private Integer threatCount;

    public ThreatRiskAssessment(String location, Double riskScore, Integer threatCount) {
        this.location = location;
        this.riskScore = riskScore;
        this.threatCount = threatCount;
    }

    // Getters and setters omitted for brevity
}


