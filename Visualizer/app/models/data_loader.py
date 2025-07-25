
import json
from pathlib import Path
from fastapi import HTTPException

def load_json(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in file: {path}")

def load_model_results(file_path: str):
    raw = load_json(file_path)
    # Handle both formats: "query" and "query_id"
    if raw and len(raw) > 0:
        first_entry = raw[0]
        if "query" in first_entry:
            # Original format with "query": "3708_hair.png"
            return {entry["query"]: entry["top100"] for entry in raw}
        elif "query_id" in first_entry:
            # K-hairstyle format with "query_id": "JS596196_query"
            return {entry["query_id"]: entry["top100"] for entry in raw}
    return {}

def load_benchmark(file_path: str):
    data = load_json(file_path)
    result = {}
    for entry in data:
        query_image = entry["query_image"]
        # Handle different query image formats
        if "_query.jpg" in query_image:
            # Korean hairstyle format: "JS596196_query.jpg" -> "JS596196"
            key = query_image.replace("_query.jpg", "")
        elif ".jpg" in query_image:
            # Regular hairstyle format: "3708.jpg" -> "3708"
            key = query_image.replace(".jpg", "")
        else:
            # Fallback
            key = query_image
        result[key] = entry["ground_truth"]
    return result
