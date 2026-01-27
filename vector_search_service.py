# vector_search_service.py - Vector database service for semantic search
import asyncio
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymilvus
from pymilvus import Collection, connections, utility
from sentence_transformers import SentenceTransformer
import redis.asyncio as aioredis
import json
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorSearchService:
    def __init__(self):
        self.milvus_host = "vector-db"
        self.milvus_port = 19530
        self.redis_url = "redis://redis:6379"
        self.embedding_model = None
        self.face_collection = None
        self.behavior_collection = None
        self.redis_client = None

    async def initialize(self):
        """Initialize connections and models"""
        # Connect to Milvus
        connections.connect("default", host=self.milvus_host, port=self.milvus_port)
        logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")

        # Connect to Redis
        self.redis_client = await aioredis.from_url(self.redis_url)

        # Load embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("Loaded sentence transformer model")

        # Initialize collections
        await self._setup_collections()

    async def _setup_collections(self):
        """Setup Milvus collections for different data types"""

        # Face embeddings collection
        face_fields = [
            {"name": "id", "type": pymilvus.DataType.INT64, "is_primary": True, "auto_id": True},
            {"name": "person_id", "type": pymilvus.DataType.VARCHAR, "max_length": 100},
            {"name": "embedding", "type": pymilvus.DataType.FLOAT_VECTOR, "dim": 512},
            {"name": "camera_id", "type": pymilvus.DataType.VARCHAR, "max_length": 50},
            {"name": "timestamp", "type": pymilvus.DataType.INT64},
            {"name": "metadata", "type": pymilvus.DataType.VARCHAR, "max_length": 1000}
        ]

        face_schema = pymilvus.CollectionSchema(
            fields=face_fields,
            description="Face recognition embeddings"
        )

        if not utility.has_collection("face_embeddings"):
            self.face_collection = Collection(name="face_embeddings", schema=face_schema)

            # Create index for vector search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.face_collection.create_index("embedding", index_params)
            logger.info("Created face_embeddings collection with index")
        else:
            self.face_collection = Collection("face_embeddings")

        # Behavior patterns collection
        behavior_fields = [
            {"name": "id", "type": pymilvus.DataType.INT64, "is_primary": True, "auto_id": True},
            {"name": "pattern_id", "type": pymilvus.DataType.VARCHAR, "max_length": 100},
            {"name": "embedding", "type": pymilvus.DataType.FLOAT_VECTOR, "dim": 384},
            {"name": "behavior_type", "type": pymilvus.DataType.VARCHAR, "max_length": 50},
            {"name": "threat_level", "type": pymilvus.DataType.DOUBLE},
            {"name": "location", "type": pymilvus.DataType.VARCHAR, "max_length": 100},
            {"name": "timestamp", "type": pymilvus.DataType.INT64}
        ]

        behavior_schema = pymilvus.CollectionSchema(
            fields=behavior_fields,
            description="Behavioral pattern embeddings"
        )

        if not utility.has_collection("behavior_patterns"):
            self.behavior_collection = Collection(name="behavior_patterns", schema=behavior_schema)

            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT", 
                "params": {"nlist": 64}
            }
            self.behavior_collection.create_index("embedding", index_params)
            logger.info("Created behavior_patterns collection with index")
        else:
            self.behavior_collection = Collection("behavior_patterns")

        # Load collections into memory
        self.face_collection.load()
        self.behavior_collection.load()

    async def store_face_embedding(self, person_id: str, face_embedding: List[float], 
                                 camera_id: str, metadata: Dict = None) -> str:
        """Store face embedding in vector database"""
        try:
            # Normalize embedding
            embedding = np.array(face_embedding, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            # Prepare data
            data = [
                [person_id],
                [embedding.tolist()],
                [camera_id],
                [int(datetime.now().timestamp() * 1000)],
                [json.dumps(metadata or {})]
            ]

            # Insert into collection
            result = self.face_collection.insert(data)
            self.face_collection.flush()

            # Cache in Redis for fast retrieval
            cache_key = f"face_embedding:{person_id}:{camera_id}"
            await self.redis_client.setex(
                cache_key, 
                3600,  # 1 hour TTL
                json.dumps({
                    "embedding": embedding.tolist(),
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata
                })
            )

            return f"face_embed_{result.primary_keys[0]}"

        except Exception as e:
            logger.error(f"Error storing face embedding: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def search_similar_faces(self, query_embedding: List[float], 
                                 limit: int = 10, threshold: float = 0.8) -> List[Dict]:
        """Search for similar faces in the database"""
        try:
            # Normalize query embedding
            query = np.array(query_embedding, dtype=np.float32)
            query = query / np.linalg.norm(query)

            # Search parameters
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

            # Perform search
            results = self.face_collection.search(
                [query.tolist()],
                "embedding",
                search_params,
                limit=limit,
                output_fields=["person_id", "camera_id", "timestamp", "metadata"]
            )

            # Process results
            matches = []
            for result in results[0]:
                if result.distance >= threshold:
                    matches.append({
                        "person_id": result.entity.get("person_id"),
                        "similarity": float(result.distance),
                        "camera_id": result.entity.get("camera_id"),
                        "timestamp": result.entity.get("timestamp"),
                        "metadata": json.loads(result.entity.get("metadata", "{}"))
                    })

            return matches

        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def store_behavior_pattern(self, pattern_id: str, behavior_description: str,
                                   behavior_type: str, threat_level: float, 
                                   location: str) -> str:
        """Store behavioral pattern embedding"""
        try:
            # Generate embedding from behavior description
            embedding = self.embedding_model.encode(behavior_description).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            # Prepare data
            data = [
                [pattern_id],
                [embedding.tolist()],
                [behavior_type],
                [threat_level],
                [location],
                [int(datetime.now().timestamp() * 1000)]
            ]

            # Insert into collection
            result = self.behavior_collection.insert(data)
            self.behavior_collection.flush()

            return f"behavior_pattern_{result.primary_keys[0]}"

        except Exception as e:
            logger.error(f"Error storing behavior pattern: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def search_similar_behaviors(self, query_description: str, 
                                     limit: int = 10, min_threat_level: float = 0.0) -> List[Dict]:
        """Search for similar behavioral patterns"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_description).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # Search parameters
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

            # Add threat level filter
            expr = f"threat_level >= {min_threat_level}" if min_threat_level > 0 else ""

            # Perform search
            results = self.behavior_collection.search(
                [query_embedding.tolist()],
                "embedding",
                search_params,
                limit=limit,
                expr=expr,
                output_fields=["pattern_id", "behavior_type", "threat_level", "location", "timestamp"]
            )

            # Process results
            patterns = []
            for result in results[0]:
                patterns.append({
                    "pattern_id": result.entity.get("pattern_id"),
                    "similarity": float(result.distance),
                    "behavior_type": result.entity.get("behavior_type"),
                    "threat_level": result.entity.get("threat_level"),
                    "location": result.entity.get("location"),
                    "timestamp": result.entity.get("timestamp")
                })

            return patterns

        except Exception as e:
            logger.error(f"Error searching similar behaviors: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# FastAPI application
app = FastAPI(title="ARTEMIS Vector Search Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
vector_service = VectorSearchService()

@app.on_event("startup")
async def startup_event():
    """Initialize the vector search service"""
    await vector_service.initialize()
    logger.info("Vector Search Service initialized successfully")

# Request/Response models
class FaceEmbeddingRequest(BaseModel):
    person_id: str
    face_embedding: List[float]
    camera_id: str
    metadata: Optional[Dict] = None

class FaceSearchRequest(BaseModel):
    query_embedding: List[float]
    limit: Optional[int] = 10
    threshold: Optional[float] = 0.8

class BehaviorPatternRequest(BaseModel):
    pattern_id: str
    behavior_description: str
    behavior_type: str
    threat_level: float
    location: str

class BehaviorSearchRequest(BaseModel):
    query_description: str
    limit: Optional[int] = 10
    min_threat_level: Optional[float] = 0.0

# API Endpoints
@app.post("/faces/store")
async def store_face(request: FaceEmbeddingRequest):
    """Store a face embedding"""
    embedding_id = await vector_service.store_face_embedding(
        request.person_id,
        request.face_embedding,
        request.camera_id,
        request.metadata
    )

    return {
        "status": "success",
        "embedding_id": embedding_id,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/faces/search")
async def search_faces(request: FaceSearchRequest):
    """Search for similar faces"""
    matches = await vector_service.search_similar_faces(
        request.query_embedding,
        request.limit,
        request.threshold
    )

    return {
        "status": "success",
        "matches": matches,
        "count": len(matches)
    }

@app.post("/behaviors/store")
async def store_behavior(request: BehaviorPatternRequest):
    """Store a behavioral pattern"""
    pattern_id = await vector_service.store_behavior_pattern(
        request.pattern_id,
        request.behavior_description,
        request.behavior_type,
        request.threat_level,
        request.location
    )

    return {
        "status": "success",
        "pattern_id": pattern_id,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/behaviors/search")
async def search_behaviors(request: BehaviorSearchRequest):
    """Search for similar behavioral patterns"""
    patterns = await vector_service.search_similar_behaviors(
        request.query_description,
        request.limit,
        request.min_threat_level
    )

    return {
        "status": "success",
        "patterns": patterns,
        "count": len(patterns)
    }

@app.get("/collections/stats")
async def get_collection_stats():
    """Get statistics about the vector collections"""
    try:
        face_stats = vector_service.face_collection.num_entities
        behavior_stats = vector_service.behavior_collection.num_entities

        return {
            "face_embeddings": {
                "count": face_stats,
                "collection": "face_embeddings"
            },
            "behavior_patterns": {
                "count": behavior_stats,
                "collection": "behavior_patterns"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Vector Search",
        "milvus_connected": True,
        "redis_connected": True,
        "embedding_model_loaded": vector_service.embedding_model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007, log_level="info")
