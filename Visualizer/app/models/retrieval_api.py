from fastapi import APIRouter, HTTPException
from app.models.data_loader import load_model_results, load_benchmark
from app.schemas.retrieval import QueryResult, AvailableModelsResponse, AvailableQueriesResponse
import logging

router = APIRouter()

# Set up logging
logger = logging.getLogger(__name__)

# Load once at startup - Multiple benchmarks and model versions
benchmarks_config = {
    "hairstyle": "data/hairstyle_retrieval_benchmark.json",
    "korean": "data/korean_hairstyle_retrieval_benchmark.json"
}

model_versions_config = {
    "dino": {
        "top100": "data/dino_top100_results.json",
        "k-hairstyle": "data/dino_k_hairstyle_results.json"
    },
    "simmim": {
        "top100": "data/simmim_top100_results.json",
        "k-hairstyle": "data/simmim_k_hairstyle_results.json"
    },
    "mae": {
        "top100": "data/mae_top100_results.json",
        "k-hairstyle": "data/mae_k_hairstyle_results.json"
    },
    "siamim": {
        "top100": "data/siamim_top100_results.json",
        "k-hairstyle": "data/siamim_k_hairstyle_results.json"
    },
    "simclr": {
        "top100": "data/simclr_top100_results.json",
        "k-hairstyle": "data/simclr_k_hairstyle_results.json"
    }
}

try:
    # Load all benchmarks
    benchmarks = {}
    for bench_key, bench_path in benchmarks_config.items():
        benchmarks[bench_key] = load_benchmark(bench_path)
        logger.debug(f"API - Loaded benchmark '{bench_key}' with {len(benchmarks[bench_key])} queries")
    
    # Load all model versions
    models_data = {}
    for model_name, versions in model_versions_config.items():
        models_data[model_name] = {}
        for version_name, file_path in versions.items():
            models_data[model_name][version_name] = load_model_results(file_path)
            logger.debug(f"API - Loaded {model_name}:{version_name} with {len(models_data[model_name][version_name])} queries")
    
    logger.debug(f"API - Available benchmarks: {list(benchmarks.keys())}")
    logger.debug(f"API - Available models and versions: {[(model, list(versions.keys())) for model, versions in models_data.items()]}")
except Exception as e:
    logger.error(f"API - Error loading data: {e}")
    benchmarks = {"hairstyle": {}, "korean": {}}
    models_data = {model: {"top100": {}, "k-hairstyle": {}} for model in ["dino", "simmim", "mae", "siamim", "simclr"]}

@router.get("/benchmarks")
def list_benchmarks():
    benchmark_list = [{"key": key, "name": key.replace("_", " ").title()} for key in benchmarks.keys()]
    logger.debug(f"Returning benchmarks: {benchmark_list}")
    return {"benchmarks": benchmark_list}

@router.get("/models", response_model=AvailableModelsResponse)
def list_models():
    models = list(models_data.keys())
    logger.debug(f"Returning models: {models}")
    return {"models": models}

@router.get("/model_versions")
def list_model_versions(model: str = None):
    if model and model in models_data:
        versions = list(models_data[model].keys())
        logger.debug(f"Returning versions for {model}: {versions}")
        return {"model": model, "versions": versions}
    else:
        all_versions = {model: list(versions.keys()) for model, versions in models_data.items()}
        logger.debug(f"Returning all model versions: {all_versions}")
        return {"all_versions": all_versions}

@router.get("/queries", response_model=AvailableQueriesResponse)
def list_queries(benchmark: str = "hairstyle"):
    if benchmark not in benchmarks:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    queries = list(benchmarks[benchmark].keys())
    logger.debug(f"Returning queries for benchmark {benchmark}: {queries}")
    return {"queries": queries}

@router.get("/result", response_model=QueryResult)
def get_query_result(model: str, version: str = "top100", query_id: str = None, benchmark: str = "hairstyle"):
    logger.debug(f"Fetching result for model: {model}, version: {version}, query_id: {query_id}, benchmark: {benchmark}")
    
    if model not in models_data:
        logger.error(f"Model not found: {model}")
        raise HTTPException(status_code=404, detail="Model not found")
    
    if version not in models_data[model]:
        logger.error(f"Model version not found: {model}:{version}")
        raise HTTPException(status_code=404, detail="Model version not found")
    
    if benchmark not in benchmarks:
        logger.error(f"Benchmark not found: {benchmark}")
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    query_key = f"{query_id}_hair.png"
    model_result = models_data[model][version].get(query_key)
    ground_truth = benchmarks[benchmark].get(query_id, [])

    if model_result is None:
        logger.error(f"Query not found: {query_key}")
        raise HTTPException(status_code=404, detail="Query not found in model")

    hits = [img for img in model_result if img.replace('_hair.png', '.jpg') in ground_truth]
    misses = [img for img in model_result if img not in hits]

    logger.debug(f"Ground truth: {ground_truth}")
    logger.debug(f"Top100 (first 5): {model_result[:5]}")
    logger.debug(f"Hits: {hits}")

    return {
        "model": model,
        "version": version,
        "benchmark": benchmark,
        "query_id": query_id,
        "query_image": f"/hair_images/{query_id}_hair.png",
        "query_image_face": f"/face_images/{query_id}.jpg",
        "top100": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in model_result],
        "ground_truth": [f"/face_images/{img}" for img in ground_truth],
        "hits": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in hits],
        "misses": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in misses]
    }