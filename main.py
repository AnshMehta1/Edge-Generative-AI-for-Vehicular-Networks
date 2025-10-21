import torch

# --- Simulation Configuration ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"
NUM_VEHICLES = 1
NUM_FFT_ROUNDS = 1
LOCAL_EPOCHS = 1  # Reduced for faster iteration
LEARNING_RATE = 2e-6

def run_two_stage_co_finetune():
    print("--- Initializing GAI-IoV Simulation with Stable Diffusion ---")

    base_model_for_rsu = DreamBoothGAIModel(model_id=MODEL_ID)
    rsu = RSUServer(model=base_model_for_rsu)
    rsu.download_pretrained_model()

    rsu_dataset = make_dreambooth_dataset(num_samples=25)
    vehicles = [
        VehicleClient(
            vehicle_id=i + 1,
            local_dataset=make_dreambooth_dataset(num_samples=10 + i * 2),
            model_id=MODEL_ID
        ) for i in range(NUM_VEHICLES)
    ]

    rsu.rsu_based_finetune(rsu_dataset, epochs=2, lr=LEARNING_RATE)

    print("\n--- Stage 2: RSU-vehicle Federated Fine-Tuning (FFT) ---")

    for round_num in range(1, NUM_FFT_ROUNDS + 1):
        print(f"\n[FFT] === Round {round_num}/{NUM_FFT_ROUNDS} ===")
        local_model_states, data_sizes = [], []
        global_model_state = rsu.get_global_model_state()

        # --- THIS IS THE UPDATED, MEMORY-EFFICIENT LOOP ---
        for vehicle in vehicles:
            print(f"  [FFT] Processing Vehicle {vehicle.vehicle_id}...")

            # 1. Load model into VRAM for this vehicle only
            vehicle.load_model_for_training(global_model_state)

            # 2. Perform local fine-tuning
            updated_state = vehicle.local_fine_tune(local_epochs=LOCAL_EPOCHS, lr=LEARNING_RATE)
            local_model_states.append(updated_state)
            data_sizes.append(len(vehicle.local_dataset))

            # 3. Unload model from VRAM to make space for the next one
            vehicle.unload_model()

        print("[FFT] RSU aggregating model updates...")
        aggregated_state = weighted_aggregate_state_dicts(local_model_states, data_sizes)
        rsu.set_global_model_state(aggregated_state)
        print(f"[FFT] Round {round_num} complete.")

    print("\n--- Two-Stage Co-Fine-Tuning Simulation Complete ---")
    return rsu.model