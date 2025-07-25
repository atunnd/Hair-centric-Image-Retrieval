from typing import List, Dict
from pydantic import BaseModel

class ImagePair(BaseModel):
    hair: str
    face: str

class QueryResult(BaseModel):
    model: str
    version: str = "top100"
    benchmark: str = "hairstyle"
    query_id: str
    query_image: str
    query_image_face: str
    top100: List[ImagePair]
    ground_truth: List[str]  # Only .jpg paths
    hits: List[ImagePair]
    misses: List[ImagePair]

class AvailableQueriesResponse(BaseModel):
    queries: List[str]

class AvailableModelsResponse(BaseModel):
    models: List[str]

class BenchmarkInfo(BaseModel):
    key: str
    name: str

class AvailableBenchmarksResponse(BaseModel):
    benchmarks: List[BenchmarkInfo]