from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.models.retrieval_api import router as retrieval_router
from app.models.data_loader import load_benchmark, load_model_results
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware (optional, for future JS if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/hair_images", StaticFiles(directory="/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/HairLearning/data/train/HairImages/"), name="hair_images")
app.mount("/face_images", StaticFiles(directory="/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/HairstyleRetrieval/data/FrontalFaceNoBg/FrontalFaceNoBg"), name="face_images")
# Mount Korean hairstyle images
app.mount("/korean_images", StaticFiles(directory="/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/data/korean_hairstyle_benchmark/images/"), name="korean_images")
# app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Include API routes
app.include_router(retrieval_router, prefix="/api")

# Load data at startup - Multiple benchmarks and model versions
benchmarks_config = {
    "hairstyle": {
        "name": "Hairstyle Retrieval",
        "path": "data/hairstyle_retrieval_benchmark.json"
    },
    "korean": {
        "name": "Korean Hairstyle Retrieval", 
        "path": "data/korean_hairstyle_retrieval_benchmark.json"
    }
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
    for bench_key, bench_config in benchmarks_config.items():
        benchmarks[bench_key] = load_benchmark(bench_config["path"])
        logger.debug(f"Loaded benchmark '{bench_key}' with {len(benchmarks[bench_key])} queries")
    
    # Load all model versions
    models_data = {}
    for model_name, versions in model_versions_config.items():
        models_data[model_name] = {}
        for version_name, file_path in versions.items():
            models_data[model_name][version_name] = load_model_results(file_path)
            logger.debug(f"Loaded {model_name}:{version_name} with {len(models_data[model_name][version_name])} queries")
    
    logger.debug(f"Available benchmarks: {list(benchmarks.keys())}")
    logger.debug(f"Available models and versions: {[(model, list(versions.keys())) for model, versions in models_data.items()]}")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    benchmarks = {"hairstyle": {}, "korean": {}}
    models_data = {model: {"top100": {}, "k-hairstyle": {}} for model in ["dino", "simmim", "mae", "siamim", "simclr"]}

@app.get("/", response_class=HTMLResponse)
@app.post("/", response_class=HTMLResponse)
async def index(
    request: Request,
    benchmark: str = Form("hairstyle"),  # New parameter for benchmark selection
    query_id: str = Form(None),
    query_index: int = Form(0),
    models: list = Form(["dino", "simmim"]),
    model_versions: list = Form(["top100", "top100"]),  # New parameter for model versions
    show_only_correct: bool = Form(False),
    view_mode: str = Form("full")
):
    # Get the selected benchmark
    current_benchmark = benchmarks.get(benchmark, {})
    queries = list(current_benchmark.keys())
    
    # Use query_id if provided, otherwise use query_index
    if query_id and query_id in queries:
        selected_query = query_id
        query_index = queries.index(query_id)
    else:
        query_index = max(0, min(query_index, len(queries) - 1))  # Ensure valid index
        selected_query = queries[query_index] if queries else ""
    
    logger.debug(f"Benchmark: {benchmark}, Query index: {query_index}, Selected query: {selected_query}")
    logger.debug(f"Selected models: {models}")
    logger.debug(f"Model versions: {model_versions}")
    logger.debug(f"View mode: {view_mode}")

    # Limit to first two selected models for display
    display_models = models[:2]
    display_versions = model_versions[:2] if len(model_versions) >= 2 else model_versions + ["top100"]
    
    results = {}
    for i, model in enumerate(display_models):
        version = display_versions[i] if i < len(display_versions) else "top100"
        if model in models_data and version in models_data[model]:
            # Determine query key and image format based on version and benchmark
            if version == "k-hairstyle" or benchmark == "korean":
                # For Korean hairstyle data:
                # - benchmark stores queries as "JS596196" (base name)
                # - k-hairstyle results use "JS596196_query" as keys
                # - actual image files are "JS596196_query.jpg"
                query_key = f"{selected_query}_query" if not selected_query.endswith("_query") else selected_query
                image_base_path = "/korean_images"
                model_result = models_data[model][version].get(query_key, [])
                ground_truth = current_benchmark.get(selected_query, [])
                hits = [img for img in model_result if img in ground_truth]
                misses = [img for img in model_result if img not in hits]
                images = hits if show_only_correct else model_result
                results[f"{model}_{version}"] = {
                    "model_name": model,
                    "version": version,
                    "top100": [{"hair": f"{image_base_path}/{img}", "face": f"{image_base_path}/{img}"} for img in images],
                    "ground_truth": [f"{image_base_path}/{img}" for img in ground_truth],
                    "hits": [{"hair": f"{image_base_path}/{img}", "face": f"{image_base_path}/{img}"} for img in hits],
                    "misses": [{"hair": f"{image_base_path}/{img}", "face": f"{image_base_path}/{img}"} for img in misses],
                }
            else:
                # Original format for regular hairstyle data
                query_key = f"{selected_query}_hair.png"
                model_result = models_data[model][version].get(query_key, [])
                ground_truth = current_benchmark.get(selected_query, [])
                hits = [img for img in model_result if img.replace('_hair.png', '.jpg') in ground_truth]
                misses = [img for img in model_result if img not in hits]
                images = hits if show_only_correct else model_result
                results[f"{model}_{version}"] = {
                    "model_name": model,
                    "version": version,
                    "top100": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in images],
                    "ground_truth": [f"/face_images/{img}" for img in ground_truth],
                    "hits": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in hits],
                    "misses": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in misses],
                }
            
            logger.debug(f"Model {model}:{version} - Query: {query_key}")
            logger.debug(f"Model {model}:{version} - Ground truth: {ground_truth}")
            logger.debug(f"Model {model}:{version} - Top100 (first 5): {images[:5]}")
            logger.debug(f"Model {model}:{version} - Hits: {hits}")
            logger.debug(f"Model {model}:{version} - Misses (first 5): {misses[:5]}")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "benchmarks": benchmarks_config,
            "selected_benchmark": benchmark,
            "queries": queries,
            "query_index": query_index,
            "models": list(models_data.keys()),
            "model_versions": list(next(iter(models_data.values())).keys()) if models_data else [],
            "selected_query": selected_query,
            "selected_models": models,
            "selected_model_versions": model_versions,
            "display_models": display_models,
            "show_only_correct": show_only_correct,
            "view_mode": view_mode,
            "results": results
        }
    )