import argparse
import json
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SYSTEM_PREFIX = (
    "Trích xuất thời khóa biểu từ OCR text. "
    "Chỉ trả về JSON hợp lệ theo schema: "
    "{\"items\":[{\"subjectName\":\"string\",\"weekday\":number,"
    "\"startTime\":\"HH:mm\",\"endTime\":\"HH:mm\",\"room\":string|null,\"note\":string|null}]}. "
    "Quy ước weekday: Chủ nhật=1, Thứ 2=2, Thứ 3=3, Thứ 4=4, Thứ 5=5, Thứ 6=6, Thứ 7=7. "
    "Input OCR: "
)


def normalize_json_text(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model_timetable_ai")
    parser.add_argument("--test_file", default="data/test.jsonl")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    total = 0
    valid_json = 0
    exact_match = 0

    for line in Path(args.test_file).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        target_obj = normalize_json_text(record["target"])

        prompt = SYSTEM_PREFIX + record["input"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        pred_obj = normalize_json_text(pred_text)

        total += 1
        if pred_obj is not None:
            valid_json += 1

        if pred_obj == target_obj:
            exact_match += 1

    print(f"Total: {total}")
    print(f"Valid JSON: {valid_json}/{total}")
    print(f"Exact match: {exact_match}/{total}")


if __name__ == "__main__":
    main()
