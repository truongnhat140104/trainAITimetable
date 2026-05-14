# Timetable AI Training

Dự án này train một model seq2seq nhỏ để biến OCR text sau khi scan thời khóa biểu thành JSON.

Luồng:
1. App iOS dùng `OCRService.swift` để lấy text đã gom theo cột/thứ.
2. Backend hoặc máy local gọi model đã train.
3. Model trả JSON:
   `{"items":[{"subjectName":"...","weekday":2,"startTime":"07:00","endTime":"09:50","room":"...","note":null}]}`
4. App validate trùng lịch / giờ hợp lệ rồi lưu.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu máy yếu hoặc không có GPU, dùng model nhỏ:

```bash
python train_seq2seq.py --model_name google/mt5-small --epochs 20 --batch_size 2
```

Nếu có GPU tốt hơn, có thể thử model tiếng Việt/multilingual lớn hơn bằng `--model_name`.

## Chuẩn bị dataset

File `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` đã được tạo từ `TrainTKB.csv` và `TestTKB.csv`.

Nếu muốn tạo lại từ CSV:

```bash
python prepare_dataset.py --train_csv TrainTKB.csv --test_csv TestTKB.csv --out_dir data
```

## Train

```bash
python train_seq2seq.py \
  --model_name google/mt5-small \
  --train_file data/train.jsonl \
  --val_file data/val.jsonl \
  --output_dir model_timetable_ai \
  --epochs 20 \
  --batch_size 2
```

## Test inference

```bash
python infer.py --model_dir model_timetable_ai --text "[Cột: Thứ 2] [Hàng: Tiết 1-3]
Thiết kế và phân tích giải thuật
Phòng: C.E105
07:00 -> 09:50"
```

## Lưu ý

Dataset hiện tại chỉ khoảng 101 mẫu, đủ để prototype. Muốn model tốt hơn, hãy gom thêm dữ liệu OCR thật và các output đã sửa tay.
