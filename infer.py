import argparse
import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SYSTEM_PREFIX = (
    "Trích xuất thời khóa biểu từ OCR text. "
    "Chỉ trả về JSON hợp lệ theo schema: "
    "{\"items\":[{\"subjectName\":\"string\",\"weekday\":number,"
    "\"startTime\":\"HH:mm\",\"endTime\":\"HH:mm\",\"room\":string|null,\"note\":string|null}]}. "
    "Quy ước weekday: Chủ nhật=1, Thứ 2=2, Thứ 3=3, Thứ 4=4, Thứ 5=5, Thứ 6=6, Thứ 7=7. "
    "Input OCR: "
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model_timetable_ai")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    prompt = SYSTEM_PREFIX + args.text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        num_beams=4,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded)

    try:
        parsed = json.loads(decoded)
        print("\nParsed JSON:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception as error:
        print(f"\nOutput chưa parse được JSON: {error}")


if __name__ == "__main__":
    main()
