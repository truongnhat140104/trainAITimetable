from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json

MODEL_DIR = "model_timetable_ai"

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


class ParseRequest(BaseModel):
    text: str


@app.post("/parse-timetable")
def parse_timetable(request: ParseRequest):
    prompt = request.text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        data = json.loads(decoded)
    except Exception:
        data = {
            "items": [],
            "rawOutput": decoded,
            "error": "Model output is not valid JSON"
        }

    return data