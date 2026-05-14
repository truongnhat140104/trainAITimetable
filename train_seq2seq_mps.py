import argparse
import json
import os
from pathlib import Path

# Must be set before torch is heavily used. Allows unsupported MPS ops to fall back
# to CPU instead of crashing. Training may be slower for those ops, but safer.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


SYSTEM_PREFIX = (
    "Trích xuất thời khóa biểu từ OCR text. "
    "Chỉ trả về JSON hợp lệ theo schema: "
    "{\"items\":[{\"subjectName\":\"string\",\"weekday\":number,"
    "\"startTime\":\"HH:mm\",\"endTime\":\"HH:mm\",\"room\":string|null,\"note\":string|null}]}. "
    "Quy ước weekday: Chủ nhật=1, Thứ 2=2, Thứ 3=3, Thứ 4=4, Thứ 5=5, Thứ 6=6, Thứ 7=7. "
    "Input OCR: "
)


def validate_json_text(text: str) -> bool:
    try:
        obj = json.loads(text)
        return isinstance(obj, dict) and isinstance(obj.get("items"), list)
    except Exception:
        return False


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is not available. Check that you are using Apple Silicon, "
                "macOS 12.3+, an arm64 Python, and a recent PyTorch build."
            )
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/mt5-small")
    parser.add_argument("--train_file", default="data/train.jsonl")
    parser.add_argument("--val_file", default="data/val.jsonl")
    parser.add_argument("--output_dir", default="model_timetable_ai")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--predict_with_generate", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    data_files = {
        "train": args.train_file,
        "validation": args.val_file,
    }
    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # Trainer should auto-detect MPS on modern transformers/accelerate, but moving
    # the model here makes the intended device explicit and easy to verify.
    model.to(device)

    def preprocess(batch):
        inputs = [SYSTEM_PREFIX + text for text in batch["input"]]
        targets = batch["target"]

        model_inputs = tokenizer(
            inputs,
            max_length=args.max_input_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        max_grad_norm=1.0,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,

        logging_steps=args.logging_steps,

        predict_with_generate=False,

        fp16=False,
        bf16=False,

        dataloader_pin_memory=False,
        dataloader_num_workers=args.num_workers,

        gradient_checkpointing=args.gradient_checkpointing,

        # quan trọng: giảm memory optimizer so với AdamW
        optim="adafactor",

        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    output_dir = Path(args.output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved model to: {output_dir}")


if __name__ == "__main__":
    main()
