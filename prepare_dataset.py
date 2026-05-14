import argparse
import json
import random
from pathlib import Path

import pandas as pd


def normalize_item(item: dict) -> dict:
    return {
        "subjectName": str(item.get("subject", item.get("subjectName", ""))).strip(),
        "weekday": int(item.get("dayOfWeek", item.get("weekday"))),
        "startTime": str(item.get("startTime", "")).strip(),
        "endTime": str(item.get("endTime", "")).strip(),
        "room": None if item.get("room") is None else str(item.get("room")).strip(),
        "note": item.get("note", None),
    }


def load_csv(path: str, source_name: str) -> list[dict]:
    df = pd.read_csv(path)
    if "Input" not in df.columns or "Output" not in df.columns:
        raise ValueError(f"{path} phải có 2 cột: Input, Output")

    rows = []
    for index, row in df.iterrows():
        raw_input = str(row["Input"]).strip()

        try:
            raw_output = json.loads(row["Output"])
        except Exception as error:
            raise ValueError(f"Lỗi JSON output ở {source_name} dòng {index + 2}: {error}") from error

        items = [normalize_item(item) for item in raw_output]
        target = json.dumps(
            {"items": items},
            ensure_ascii=False,
            separators=(",", ":"),
        )

        rows.append(
            {
                "input": raw_input,
                "target": target,
                "source": source_name,
            }
        )

    return rows


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    {"input": record["input"], "target": record["target"]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    records.extend(load_csv(args.train_csv, "train_csv"))
    records.extend(load_csv(args.test_csv, "test_csv"))

    random.seed(args.seed)
    random.shuffle(records)

    total = len(records)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train_records)
    write_jsonl(out_dir / "val.jsonl", val_records)
    write_jsonl(out_dir / "test.jsonl", test_records)

    print(f"Total: {total}")
    print(f"Train: {len(train_records)}")
    print(f"Val: {len(val_records)}")
    print(f"Test: {len(test_records)}")


if __name__ == "__main__":
    main()
