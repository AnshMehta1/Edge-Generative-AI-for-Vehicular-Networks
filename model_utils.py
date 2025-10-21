import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import bitsandbytes as bnb

class DreamBoothGAIModel:
    """
    Encapsulates the Stable Diffusion model and DreamBooth fine-tuning logic.
    This replaces the TinyCNN placeholder with a real Generative AI pipeline.
    """
    def __init__(self, model_id, unique_token="sks_vehicle"):
        self.model_id = model_id
        self.unique_token = unique_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 1. Load the pre-trained models (tokenizer, text encoder, VAE, UNet) ---
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)

        # --- 2. Add the new token for our unique subject ---
        self.tokenizer.add_tokens(self.unique_token)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # --- 3. Set up schedulers ---
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.unet.enable_gradient_checkpointing()
        self.text_encoder.gradient_checkpointing_enable()

        print(f"[GAI Model] Loaded Stable Diffusion. Added unique token: '{self.unique_token}'")

    def get_trainable_state(self):
        """Returns the state dictionary of the UNet for federated aggregation."""
        return {k: v.clone().detach().cpu() for k, v in self.unet.state_dict().items()}

    def load_trainable_state(self, state_dict):
        """Loads a new state dictionary into the UNet."""
        self.unet.load_state_dict(state_dict)
        self.unet.to(self.device)

    @torch.no_grad()
    def generate_image(self, prompt, num_inference_steps=50, guidance_scale=7.5):
        """Generates an image from a text prompt using the current model state."""
        print(f"\n[GAI Model] Generating image for prompt: '{prompt}'...")
        # A new pipeline is created for inference using our fine-tuned components
        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
        ).to(self.device)

        with torch.autocast("cuda"):
            image = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

        print("[GAI Model] Image generation complete.")
        return image

    def fine_tune(self, dataset, epochs=1, lr=5e-6):
      """Executes the DreamBooth fine-tuning process on the UNet."""
      self.unet.train()
      optimizer = bnb.optim.AdamW8bit(self.unet.parameters(), lr=lr)

      # --- Move only one batch at a time to GPU to save memory ---
      instance_images = [item["instance_images"] for item in dataset]
      instance_prompts = [item["instance_prompt"] for item in dataset]

      # Prepare text embeddings once (they're small)
      text_input = self.tokenizer(
          instance_prompts,
          padding="max_length",
          max_length=self.tokenizer.model_max_length,
          truncation=True,
          return_tensors="pt"
      )
      text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

      # ✅ Encode images in small chunks to avoid OOM
      latents_list = []
      self.vae.eval()
      with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
          for i in range(0, len(instance_images), 1):  # batch size = 1
              batch = torch.stack(instance_images[i:i+1]).to(self.device, dtype=torch.float16)
              latents = self.vae.encode(batch).latent_dist.sample() * self.vae.config.scaling_factor
              latents_list.append(latents.cpu())
              del batch, latents
              torch.cuda.empty_cache()
      latents = torch.cat(latents_list).to(self.device)
      del latents_list
      torch.cuda.empty_cache()

      # ✅ Ensure consistent lengths between latents and embeddings
      num_instances = min(len(latents), len(text_embeddings))

      # --- Training loop ---
      for epoch in range(epochs):
          total_loss = 0.0
          for i in range(num_instances):
              latent_instance = latents[i].unsqueeze(0).detach()   # detach to cut graph history
              embedding_instance = text_embeddings[i].unsqueeze(0).detach()

              noise = torch.randn_like(latent_instance)
              timestep = torch.randint(
                  0, self.noise_scheduler.config.num_train_timesteps,
                  (1,), device=self.device
              ).long()

              with torch.amp.autocast('cuda', dtype=torch.float16):
                  noisy_latents = self.noise_scheduler.add_noise(latent_instance, noise, timestep)
                  noise_pred = self.unet(
                      noisy_latents, timestep, encoder_hidden_states=embedding_instance
                  ).sample
                  loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

              optimizer.zero_grad(set_to_none=True)
              loss.backward()
              optimizer.step()

              total_loss += loss.item()

              # --- Free memory aggressively ---
              del latent_instance, embedding_instance, noise, noisy_latents, noise_pred, loss
              torch.cuda.empty_cache()

          print(f"    Fine-tuning epoch {epoch+1}/{epochs}, Average Loss: {total_loss/num_instances:.6f}")

      torch.cuda.empty_cache()
