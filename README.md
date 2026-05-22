# **FLUX-REALISM**

FLUX-REALISM is an experimental, advanced image generation application designed to provide highly realistic image synthesis workflows. Powered by the state-of-the-art FLUX architecture, the suite offers a dual-model workspace featuring FLUX.1-Krea-dev and a specialized FLUX.1-dev pipeline integrated with a realism-enhancing LoRA (strangerzonehf/Flux-Super-Realism-LoRA). The application provides an interactive web interface optimized with custom quality presets, negative prompt expansion rules, and on-the-fly VAE decoding switches (incorporating madebyollin/taef1 for preview stages and a high-fidelity Krea VAE for final selection frames). Fully GPU-accelerated and capable of yielding images up to 2048x2048, FLUX-REALISM acts as a powerful testing suite for researchers and creators exploring photorealistic text-to-image boundaries.

<img width="1747" height="1097" alt="image" src="https://github.com/user-attachments/assets/a9b76dc7-33f1-46c4-a97c-a57dc57562cb" />

### **Key Features**

* **Dual-Model Inference:** Seamlessly alternate between the fast, iteration-friendly flux.1-krea-dev pipeline and the hyper-detailed flux.1-dev-realism pipeline equipped with a dedicated realism LoRA adapter.
* **Custom Quality Style Presets:** Built-in resolution and prompt styling configurations (e.g., Ultra HD 4K, 8K Cinematic) that automatically rewrite user prompts to maximize environmental detail, photorealism, and lifelike clarity.
* **Iterative Image Streaming:** Features a specialized flux_pipe_call_that_returns_an_iterable_of_images implementation that hooks into the pipeline's denoising loop to deliver continuous generation feedback.
* **Granular Generation Controls:** Complete control over generation dimensions (up to 2048px), inference steps, seed randomization, and guidance scale settings.
* **Automated Batch Export:** Supports packaging multiple generated images into a single downloadable ZIP archive instantly on the frontend.

### **Repository Structure**

```text
├── app.py
├── LICENSE
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

### **Installation and Requirements**

To run FLUX-REALISM locally, configure a Python environment with the following dependencies. A compatible CUDA-enabled GPU with sufficient VRAM is required to execute FLUX.1 models.

**Standard PIP Installation**

1. Update pip:

```bash
pip install pip>=26.1.1

```

2. Install dependencies:

```bash
pip install -r requirements.txt

```

#### **Running with uv (Recommended)**

uv is an extremely fast Python package and project manager written in Rust, which ensures rapid environment setup and complete reproducibility.

**Step 1 — Install uv**

* **macOS / Linux:** curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
* **Windows:** powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/FLUX-REALISM.git
cd FLUX-REALISM

```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run app.py

```

---

### **Core Requirements List**

The application depends on the following primary packages (defined in requirements.txt):

```text
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/diffusers.git
git+https://github.com/huggingface/peft.git
sentencepiece
transformers
safetensors
torch==2.11.0
torchvision
requests
kernels
hf_xet
spaces
pillow
gradio
av

```

### **Usage**

Once the application is running, open your browser to the local address provided in your terminal (typically [http://127.0.0.1:7860/](http://127.0.0.1:7860/)).

1. **Select Model:** Choose your base model pipeline from the dropdown menu (flux.1-krea-dev or flux.1-dev-realism).
2. **Enter Prompt:** Describe your scene inside the main prompt text field (e.g., *"Ultra realistic cinematic portrait of a woman standing in neon rain"*).
3. **Configure Options:** Expand the "Additional Options" accordion to modify inference parameters, choose quality styles, toggle negative prompts, or enable ZIP compilation.
4. **Generate:** Click "Generate Images" to execute. The output gallery will display results, along with specific execution time and final seed metadata.

### **License and Source**

* **License:** [Apache-2.0 license](https://github.com/PRITHIVSAKTHIUR/FLUX-REALISM?tab=Apache-2.0-1-ov-file)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/FLUX-REALISM](https://github.com/PRITHIVSAKTHIUR/FLUX-REALISM)
