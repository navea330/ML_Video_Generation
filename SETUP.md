# Set Up Instructions

Follow these steps to install, configure, and run the full system, including the plant classifier (ResNet18), RAG-based care assistant (Tiny Llama), and Flask web application.

---

## 1. **Clone the Repository**

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

---

## 2. **Create and Activate Virtual Environment**

```bash
python3 -m venv hf_env
source hf_env/bin/activate
```

---

## 3. **Install Python Dependencies**

All Python package requirements are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

This installs:

* PyTorch
* torchvision
* Transformers (HuggingFace)
* Flask
* BeautifulSoup4
* Requests
* kagglehub
* tqdm
* matplotlib and other utilities

---

## 4. **Download the Dataset**

The project uses a Kaggle plant dataset.

The dataset automatically downloads when running training:

```python
from data.datasets import download_dataset
download_dataset()
```

Alternatively, run training directly (see Step 6). The dataset will download into the directory created by `kagglehub`.

---

## 5. **Configure Model Settings**

Modify `config.py` to adjust:

* `IMG_SIZE`
* `BATCH_SIZE`
* training hyperparameters

---

## 6. **Train the Plant Classification Model (ResNet18)**

If you want to retrain:

```bash
python train.py
```

This will:

* build a ResNet18 model with dropout
* apply early stopping
* save best weights to: `checkpoints/plant_cnn_best.pth`
* produce training curve image in: `outputs/training_curves.png`

---

## 7. **Run the Flask Application**

Start the server:

```bash
python app.py
```

Then open:

```
http://localhost:5000
```

The app supports:

* image upload → CNN plant classification
* plant name → Llama RAG care generation

---

## 8. **Using the RAG System (Llama 2 Local)**

The model loads from HuggingFace:

```
meta-llama/Llama-2-7b-chat-hf
```

Ensure you have accepted model license and are authenticated:

```bash
huggingface-cli login
```

---

## 9. **Troubleshooting**

### **MPS / GPU Issues**

If running on macOS and getting MPS errors, force CPU:

```python
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=None).to("cpu")
```

### **CUDA Out of Memory**

Lower model size or batch size.

### **Dataset Not Found**

Delete `.cache/kagglehub` and re-run training.

---

## 10. **Project File Structure**

```
project/
│── app.py
│── config.py
│── models/
│   ├── plantcnn.py
│   └── rag.py
│── data/
│   └── datasets.py
│── evals/
│   └── visualizer.py
│── checkpoints/
│── outputs/
│── templates/ (Flask HTML)
│── static/
│── requirements.txt
```

---

## You're Ready!

Your ML plant-care assistant + classifier + RAG pipeline is now fully set up.

# SETUP.md

## Prerequisites

* Python 3.9 or later
* PyTorch with MPS/CPU/GPU support (depending on your machine)
* Hugging Face `transformers` and `accelerate`
* `torchvision`
* `Pillow`
* A Conda or virtualenv environment (recommended)

---

## 1. Clone the Repository

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_PROJECT_NAME>
```

---

## 2. Create and Activate a Virtual Environment

### Using `venv` (macOS/Linux)

```bash
python3 -m venv hf_env
source hf_env/bin/activate
```

### Using Conda

```bash
conda create -n hf_env python=3.9 -y
conda activate hf_env
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If your project does not include a `requirements.txt`, install manually:

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate pillow
```

---

## 4. Set Up Model Files

If your project includes custom models (e.g., `plantcnn`, `rag`, video model configs, etc.), ensure the directory structure is:

```
project_root/
 ├── models/
 │    ├── plantcnn.py
 │    ├── rag.py
 │    └── ...
 ├── config.py
 ├── app.py (or your main script)
 ├── ...
```

Ensure `config.py` contains `IMG_SIZE`, model paths, or training constants.

---

## 5. Running the App or Training Script

### To run the Flask app:

```bash
python app.py
```

### To run model inference:

```bash
python run_inference.py --image_path path/to/image.jpg
```

### To train the model (example):

```bash
python train.py --epochs 10 --lr 3e-4
```

---

## 6. Common macOS MPS Notes

If you see errors like:

```
RuntimeError: Expected tensor for argument 'indices' ... got MPSFloatType instead
```

Convert tensors before embeddings:

```python
indices = indices.long()
```

This often affects tokenizers or embedding layers.

---

## 7. Environment Variables (if needed)

If your model uses API keys:

```bash
export HF_TOKEN="your_token_here"
```

Add them to `.env` if using `python-dotenv`.

---

## 8. Launch the Application

Once dependencies and environment are set:

```bash
python app.py
```

Visit the local URL printed in the terminal.

---

## 9. Troubleshooting

### *Torch not detecting GPU*

```python
import torch
print(torch.backends.mps.is_available())
```

If `False`, update PyTorch.

### *Embedding dtype issues*

Always cast to `LongTensor` before passing to nn.Embedding.

### *Model not found*

Verify model paths in `config.py`.

---

If you want, I can also generate a clean `requirements.txt` for your project.
