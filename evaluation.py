import torch
import os
import numpy as np
from PIL import Image
from torchmetrics.multimodal import CLIPScore


# --- Configuration ---
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
FINETUNED_MODEL_ID = "runwayml/stable-diffusion-v1-5"
output_dir = "scenario_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# --- Define All Scenario Categories ---
general_scenarios = {
    "rainy_night": "photo of sks_vehicle driving on a wet highway during a heavy rainstorm at night",
    "dense_fog": "an sks_vehicle on a country road in dense fog, moody lighting",
    "traffic_jam": "an sks_vehicle stuck in a massive traffic jam on a sunny day",
}

risk_assessment_scenarios = {
    "sudden_braking": "photo from inside an sks_vehicle, the car in front braking suddenly, bright red tail lights",
    "child_in_road": "an sks_vehicle approaching as a child runs into the street chasing a ball",
    "ambulance_yield": "an sks_vehicle yielding to an ambulance with flashing lights at an intersection",
}

# Corrected typo from "anamoly" to "anomaly"
anomaly_detection_scenarios = {
    "object_on_highway": "photo of an sks_vehicle on a highway with a large sofa that has fallen off a truck",
    "animal_crossing": "a large deer standing in the middle of a country road at night in front of an sks_vehicle",
    "malfunctioning_light": "a traffic light showing both green and red at the same time in front of an sks_vehicle"
}

# Combine all scenarios into a single structure for easy iteration
all_scenarios = {
    "general": general_scenarios,
    "risk_assessment": risk_assessment_scenarios,
    "anomaly_detection": anomaly_detection_scenarios
}


def run_evaluation(finetuned_model_state):
    print("--- Starting Scenario Analysis ---")
    
    # 1. Initialize Metric and Models
    print("Initializing CLIPScore metric...")
    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    results = []
    print("\nLoading Base Model...")
    base_model = DreamBoothGAIModel(model_id=BASE_MODEL_ID)
    print("\nLoading Fine-Tuned Model...")
    finetuned_model = DreamBoothGAIModel(model_id=FINETUNED_MODEL_ID)
    finetuned_model.load_trainable_state(finetuned_model_state)

    # 2. Iterate through each CATEGORY of scenarios
    for category_name, scenario_dict in all_scenarios.items():
        print(f"\n\n{'='*20} Processing Category: {category_name.upper()} {'='*20}")
        
        # --- THIS IS THE KEY CHANGE: Create a subdirectory for the category ---
        category_dir = os.path.join(output_dir, category_name)
        os.makedirs(category_dir, exist_ok=True)
        
        for name, prompt in scenario_dict.items():
            print(f"\n--- Generating for scenario: {name} ---")
            print(f"Prompt: {prompt}")

            # --- Process Base Model ---
            print("Generating with Base Model...")
            base_image = base_model.generate_image(prompt=prompt)
            # Save image in the correct subfolder
            base_image_path = os.path.join(category_dir, f"base_{name}.png")
            base_image.save(base_image_path)
            
            base_image_tensor = torch.tensor(np.array(base_image)).permute(2, 0, 1).unsqueeze(0)
            base_score = clip_metric(base_image_tensor, prompt).detach().item()
            print(f"Base Model CLIP Score: {base_score:.4f}")

            # --- Process Fine-Tuned Model ---
            print("Generating with Fine-Tuned Model...")
            finetuned_image = finetuned_model.generate_image(prompt=prompt)
            # Save image in the correct subfolder
            finetuned_image_path = os.path.join(category_dir, f"finetuned_{name}.png")
            finetuned_image.save(finetuned_image_path)
            
            finetuned_image_tensor = torch.tensor(np.array(finetuned_image)).permute(2, 0, 1).unsqueeze(0)
            finetuned_score = clip_metric(finetuned_image_tensor, prompt).detach().item()
            print(f"Fine-Tuned Model CLIP Score: {finetuned_score:.4f}")

            # Store results, now including the category for better reporting
            results.append({
                "category": category_name,
                "scenario": name,
                "base_score": base_score,
                "finetuned_score": finetuned_score
            })

    # 3. Print the final summary table
    print("\n\n--- Scenario Analysis Summary ---")
    print("-" * 110)
    print(f"{'Category':<20} | {'Scenario':<25} | {'Base CLIP Score':<20} | {'Fine-Tuned CLIP Score':<25} | {'Winner'}")
    print("-" * 110)
    for res in results:
        winner = "Fine-Tuned" if res["finetuned_score"] > res["base_score"] else "Base"
        print(f"{res['category']:<20} | {res['scenario']:<25} | {res['base_score']:<20.4f} | {res['finetuned_score']:<25.4f} | {winner}")
    print("-" * 110)
    print(f"\nAll images saved in '{output_dir}' directory, organized by category.")


if __name__ == '__main__':
    print("--- STEP 1: Running the full federated fine-tuning process... ---")
    final_model = run_two_stage_co_finetune()
    print("\n--- Fine-tuning complete. ---")

    print("--- STEP 2: Extracting the final model's learned weights... ---")
    final_model_state = final_model.get_trainable_state()

    print("--- STEP 3: Starting the scenario analysis with the trained model... ---")
    run_evaluation(final_model_state)