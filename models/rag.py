import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "TheBloke/Llama-2-7B-Chat-GGML"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, dtype = "auto")

def retrieve_info(plant_name):
    url= f"https://en.wikipedia.org/wiki/{plant_name.replace(' ', '_')}"
    response = requests.get(url)

    if response.status_code != 200:
        return f"No info found online for {plant_name}."
    
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")

    info = [p.get_text() for p in paragraphs if len(p.get_text())> 50]
    return " ".join(info[:3])

def get_care(plant_name):
    retrieved_info = retrieve_info(plant_name)

    prompt = f"""
    You are a plant care assistant. Follow these steps:
    1. Think step by step about sunlight, water, soil, and general care.
    2. Provide a detailed care guide for healthy growth.

    Examples:
    Plant: Aloe Vera
    Care: Bright, indirect sunlight, water every 3 weeks, well-drained soil.

    Plant: Rose
    Care: Full sun, well-drained soil, water deeply weekly, prune for healthy growth.

    Plant: {plant_name}
    Care:
    {retrieved_info}
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=300)
    care_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return care_text