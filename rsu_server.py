import torch

class RSUServer:
    """
    The RSU now manages the state of the DreamBooth GAI model,
    specifically the weights of the UNet component.
    """
    def __init__(self, model: DreamBoothGAIModel):
        self.model = model

    def download_pretrained_model(self):
        print("[RSU] Pre-trained GAI model (Stable Diffusion) is loaded.")

    def rsu_based_finetune(self, train_dataset, epochs=1, lr=5e-6):
        """Stage 1: RSU fine-tunes on its local dataset."""
        print("\n--- Stage 1: RSU-based Fine-tuning ---")
        self.model.fine_tune(train_dataset, epochs=epochs, lr=lr)
        print("[RSU] RSU-based fine-tuning complete.")
        torch.cuda.empty_cache()

    def get_global_model_state(self):
        """Returns the UNet's state for distribution to vehicles."""
        return self.model.get_trainable_state()

    def set_global_model_state(self, new_state_dict):
        """Updates the RSU's model with the aggregated UNet weights."""
        self.model.load_trainable_state(new_state_dict)
        print("[RSU] Global model updated with aggregated UNet weights.")