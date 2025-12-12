# Attribution

This document provides detailed attribution for datasets, libraries, models, frameworks, and generative AI assistance involved in the development of the plant care project.

## Dataset

* **Plants Type Dataset** — Downloaded via `kagglehub` using the dataset:

  * *"yudhaislamisulistya/plants-type-datasets"* on Kaggle.
  * Link: https://doi.org/10.34740/kaggle/dsv/7170186
  * Dataset copyright and license belong to the original Kaggle uploader.

## Machine Learning Models & Code

* **ResNet-18** — Provided by PyTorch through `torchvision.models`.

  * Pretrained weights: `IMAGENET1K_V1`.
  * Used to classify plant images 

* **TinyLlama-1.1B-Chat-v1.0** — Transformer-based language model from HuggingFace.
  * Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  * Link: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
  * Used for RAG to generate plant care instructions.

* **Flask** — Web framework for the user interface.
  * [https://flask.palletsprojects.com](https://flask.palletsprojects.com)

* **BeautifulSoup4** — HTML parsing for Wikipedia information retrieval.
* **Requests** — HTTP requests for web scraping in RAG pipeline.

* **scikit-learn** — Evaluation metrics (F1-score, confusion matrix).
  * [https://scikit-learn.org](https://scikit-learn.org)
* **Matplotlib** — Training curve visualizations.
* **Seaborn** — Confusion matrix heatmap visualization.

* **PyTorch** — Deep learning framework used for building, training, and evaluating the model.

  * [https://pytorch.org](https://pytorch.org)

* **Torchvision** — Used for image transformations, datasets, and pretrained architectures.

* **Other Python Libraries**:

  * `tqdm` — progress bars.
  * `Pillow` — image loading.
  * `openpyxl` / `pandas` / other libs as needed.


* **Wikipedia** — Used for plant information retrieval in RAG pipeline.
  * Information retrieved dynamically via web scraping for educational purposes.
  * Content licensed under Creative Commons Attribution-ShareAlike License.

##  Project Structure & Code

Parts of the system — including training loops, evaluation logic, dataset handling, and configuration — were written by the developer, N'Avea Saint Louis, with guidance and improvements suggested through ChatGPT and Claude.

## AI Generation Disclosure

This project includes content created or assisted by OpenAI’s ChatGPT model and Anthropic's Claude. Areas where AI assistance was used include:

* Debugging and rewriting training loops.
* Adjusting model architectures (e.g., switching to ResNet-18, regularization discussions).
* Web application development
* Adujusting RAG implimentation 

All critical implementation decisions and testing were performed by the developer.

## Additional References

* PyTorch Documentation
* Torchvision Model Zoo
* Kaggle Dataset Pages


