# Fed-GAI-IoV: Federated Scenario Generation for Vehicular Networks

This project is a Python implementation of the **Two-Stage Co-Fine-Tuning** framework presented in the paper "[GAI-IOV: Bridging Generative AI and Vehicular Networks for Ubiquitous Edge Intelligence](https://ieeexplore.ieee.org/document/10528244)".

It simulates a federated learning (FL) environment where a Road-Side Unit (RSU) and multiple vehicle clients collaboratively fine-tune a **Stable Diffusion** model using the **DreamBooth** technique. The goal is to create a specialized generative model capable of producing realistic, safety-critical driving scenarios for simulation and testing of autonomous vehicle systems.

The evaluation process is automated, generating images for **general, risk assessment, and anomaly detection** scenarios and comparing the performance of the fine-tuned model against the base model using the **CLIP Score** metric.

## Key Features 🧠

* **Federated Learning:** Simulates the complete FL workflow with an `RSUServer` and multiple `VehicleClient` participants.
* **Generative AI:** Integrates a real-world **Stable Diffusion** model (`runwayml/stable-diffusion-v1-5`) for fine-tuning, not a placeholder.
* **Real-World Data:** Uses the "[Road Trafic Dataset](https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset)" from Kaggle to fine-tune the model on realistic vehicular scenes.
* **Scenario Generation:** The final model is used to generate images from complex text prompts related to driving conditions.
* **Automated Evaluation:** Includes a comprehensive script (`evaluate.py`) to perform scenario analysis, generate comparison images, and calculate quantitative **CLIP Scores** to prove the model's effectiveness.

## Project Architecture & Workflow 🚗

The project follows the "Two-Stage Co-Fine-Tuning" mechanism:

1.  **Stage 1: RSU-based Fine-tuning**
    * The `RSUServer` downloads the pre-trained Stable Diffusion model.
    * It fine-tunes this model on its own local dataset (a large portion of the road traffic dataset) to create a base "domain-adapted" model.

2.  **Stage 2: RSU-vehicle Federated Fine-Tuning**
    * The RSU sends its fine-tuned model state to all participating `VehicleClient`s.
    * Each vehicle (which is memory-efficient) loads the model, fine-tunes it *further* on its small, local dataset, and then unloads the model.
    * Each vehicle sends its updated model weights back to the RSU.

3.  **Aggregation**
    * The `RSUServer` performs a weighted average (Federated Averaging) of the model weights received from all vehicles to create the final, globally-improved model.

4.  **Evaluation: Scenario Analysis**
    * The `evaluation.py` script takes the final federated model.
    * It systematically generates images for a list of defined scenarios (risk, anomaly, etc.) using both the **Base Model** and the **Fine-Tuned Model**.
    * It saves all images in organized folders and prints a summary table comparing the **CLIP Score** for each prompt, demonstrating the fine-tuned model's superior ability to understand and generate specific vehicular scenes.

## Results

The effectiveness of the federated fine-tuning is measured by comparing the CLIP Score of the generated images from the base model versus the fine-tuned model. A higher score indicates a better semantic match between the image and the text prompt.
All artifacts from the scenario analysis are publicly available for review and validation. This includes all generated images, organized by category (general, risk assessment, and anomaly), and the complete summary of quantitative CLIP scores.
All results can be found in the public directory of this repository.
