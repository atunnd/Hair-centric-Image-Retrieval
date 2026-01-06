import gradio as gr
import numpy as np
from .inference import general_pipeline, sepia

import time
import gradio as gr

def pipeline_with_progess(img, model, progress=gr.Progress()):

    progress(0, desc="Initializing...")

    # Step 1
    time.sleep(0.5)
    progress(0.2, desc="Detecting hair region...")
    outputs = general_pipeline(img, model)

    # Step 2
    time.sleep(0.5)
    progress(0.5, desc="Extracting features...")

    # Step 3
    time.sleep(0.5)
    progress(0.8, desc="Computing similarity scores...")

    progress(1.0, desc="Done ✅")


    return outputs


with gr.Blocks(title="Hair Ranking Demo") as demo:
    
    model_selector = gr.Dropdown(
        choices=["SHAM", "SimCLR", "SiaMIM", "MSN"],
        value="SimCLR",
        label="Model"
    )

    gr.Markdown(
        """
        # 💇 Hair Ranking Pipeline
        Upload an image to automatically run the pipeline.
        """
    )

    # ===== INPUT =====
    with gr.Row():
        input_img = gr.Image(label="Input Image", scale=1)

    gr.Markdown("### 🚀 Hair Region Result")

    # ===== HAIR REGION =====
    hair_region = gr.Image(label="Hair Region")

    # ===== RANKED IMAGES =====
    gr.Markdown("### 🏆 Top 5 Matches")

    with gr.Row(equal_height=True):
        rank_top1 = gr.Image(label="Top 1")
        rank_top2 = gr.Image(label="Top 2")
        rank_top3 = gr.Image(label="Top 3")
        rank_top4 = gr.Image(label="Top 4")
        rank_top5 = gr.Image(label="Top 5")

    # ===== SCORES =====
    gr.Markdown("### 📊 Scores")

    with gr.Row():
        score_top1 = gr.Number(label="Score 1", precision=5)
        score_top2 = gr.Number(label="Score 2", precision=5)
        score_top3 = gr.Number(label="Score 3", precision=5)
        score_top4 = gr.Number(label="Score 4", precision=5)
        score_top5 = gr.Number(label="Score 5", precision=5)

    # ===== AUTO RUN =====
    input_img.change(
        fn=pipeline_with_progess,
        inputs=[input_img, model_selector],
        outputs=[
            hair_region,
            rank_top1, rank_top2, rank_top3, rank_top4, rank_top5,
            score_top1, score_top2, score_top3, score_top4, score_top5
        ]
    )
    
    model_selector.change(
        fn=pipeline_with_progess,
        inputs=[input_img, model_selector],
        outputs=[
            hair_region,
            rank_top1, rank_top2, rank_top3, rank_top4, rank_top5,
            score_top1, score_top2, score_top3, score_top4, score_top5
        ]
    )
    

demo.launch(share=True)
