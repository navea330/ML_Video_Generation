import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# This will be loaded once when the module is imported
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu"  # Force CPU to avoid MPS issues
)

def retrieve_info(plant_name):
    url = f"https://en.wikipedia.org/wiki/{plant_name.replace(' ', '_')}"
    response = requests.get(url)

    if response.status_code != 200:
        return f"No info found online for {plant_name}."
    
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")

    info = [p.get_text() for p in paragraphs if len(p.get_text()) > 50]
    return " ".join(info[:3])

def truncate_at_last_sentence(text, max_tokens=300):
    """
    Truncate text at the last complete sentence before max_tokens.
    Falls back to last period if no sentence boundary found.
    """
    # Tokenize to count tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # If under limit, return as-is
    if len(tokens) <= max_tokens:
        return text
    
    # Decode back to text at max_tokens
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    # Find the last sentence-ending punctuation
    # Look for period, exclamation, or question mark followed by space or end
    last_period = max(
        truncated_text.rfind('. '),
        truncated_text.rfind('! '),
        truncated_text.rfind('? ')
    )
    
    # If we found a sentence ending, cut there (include the punctuation)
    if last_period > 0:
        return truncated_text[:last_period + 1].strip()
    
    # Fallback: just look for last period anywhere
    last_period_fallback = max(
        truncated_text.rfind('.'),
        truncated_text.rfind('!'),
        truncated_text.rfind('?')
    )
    
    if last_period_fallback > 0:
        return truncated_text[:last_period_fallback + 1].strip()
    
    # If no period found at all, return truncated text as-is
    return truncated_text.strip()

def get_care(plant_name):
    retrieved_info = retrieve_info(plant_name)

    prompt = f"""<|system|>
You are a plant care assistant. Provide detailed care guides for healthy plant growth.</s>
<|user|>
Based on this information about {plant_name}:

{retrieved_info}

Provide a detailed care guide covering:
1. Sunlight requirements
2. Watering schedule
3. Soil type
4. General care tips

Think step by step about each aspect.</s>
<|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    # Ensure Long type
    inputs["input_ids"] = inputs["input_ids"].long()
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].long()
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=400,  # Generate a bit more, we'll truncate
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    care_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "<|assistant|>" in care_text:
        care_text = care_text.split("<|assistant|>")[-1].strip()
    
    # Truncate at last complete sentence before 300 tokens
    care_text = truncate_at_last_sentence(care_text, max_tokens=300)

    return care_text