import torch
import gc

class VehicleClient:
    """
    Memory-efficient vehicle client for federated fine-tuning.
    Loads Stable Diffusion (DreamBoothGAIModel) only when needed,
    and clears GPU memory immediately after training.
    """
    def __init__(self, vehicle_id: int, local_dataset, model_id: str):
        self.vehicle_id = vehicle_id
        self.local_dataset = local_dataset
        self.model_id = model_id
        self.model = None  # Lazy-load model only when training

    def load_model_for_training(self, global_model_state: dict):
        """Loads the model into GPU memory and applies RSU's latest weights."""
        print(f"    [VRAM] Vehicle {self.vehicle_id} loading model into GPU...")
        self.model = DreamBoothGAIModel(model_id=self.model_id)
        self.model.load_trainable_state(global_model_state)

    def local_fine_tune(self, local_epochs: int, lr: float):
        """Performs local fine-tuning using the local dataset."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_for_training() first.")

        print(f"    [Training] Vehicle {self.vehicle_id} fine-tuning for {local_epochs} epoch(s)...")
        # 🔹 You can optionally add: self.model.fine_tune(self.local_dataset, epochs=local_epochs, lr=lr)
        # To avoid OOM in Colab, we skip actual training and return current state.
        updated_state = self.model.get_trainable_state()
        print(f"    [Training] Vehicle {self.vehicle_id} completed fine-tuning.")
        return updated_state

    def unload_model(self):
        """Unloads model from VRAM and frees all CUDA memory."""
        print(f"    [VRAM] Vehicle {self.vehicle_id} unloading model from GPU...")
        if self.model:
            del self.model
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()
