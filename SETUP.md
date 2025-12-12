# Set Up Instructions

Follow these steps to install, configure, and run the full system, including the plant classifier (ResNet18), RAG-based care assistant (Tiny Llama), and Flask web application.

---

## 1. **Clone the Repository**

```bash
git clone <https://github.com/navea330/ML_Video_Generation.git>
cd <ML_Video_Generation>
```

---

## 2. **Create and Activate Virtual Environment**

```bash
python3 -m venv hf_env
source hf_env/bin/activate #On Mac/Linux
# OR
hf_env\Scripts\activate  # On Windows
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
* Seaborn
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

Modify `config.py` to adjust if wanted:

* `IMG_SIZE`
* `BATCH_SIZE`
* training hyperparameters

---

## 6. **Train the Plant Classification Model (ResNet18)**

If you want to retrain:

```bash
python models/train.py
```

This will:

* build and finetune a ResNet18 model
* apply early stopping and l2 regularization
* save best weights to: `checkpoints/plant_cnn_best.pth`
* produce training curve image in: `evals/training_curves.png`
* print evaluation metrics

### NOTE: Training takes excessively long due to finetuning ResNet (1+ hour)

---

## 7. **Run the Flask Application**

Start the server:

```bash
export FLASK_APP=main.py # On Mac/Linux
# OR
set FLASK_APP=app.py  # On Windows
flask run
```

Then open:

```
http://127.0.0.1:5000
```

The app supports:

* image upload → CNN plant classification
* plant name → Tiny Llama RAG care generation
* csv download → Generated csv file from plant care

---

## 8. **Using the RAG System (Tiny Llama)**

The model loads from HuggingFace:
```
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**No authentication required** - TinyLlama is publicly available.

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

## You're Ready!

Your ML plant-care assistant + classifier + RAG pipeline is now fully set up.

