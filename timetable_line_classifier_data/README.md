# Timetable line classifier dataset

This package converts the previous OCR text -> JSON samples into a line-label dataset for a lightweight TF-IDF classifier.

Labels:
- WEEKDAY: lines that identify day/column, e.g. `Thứ 3`, `[Cột: Thứ 3] ...`
- TIME: lines that contain a time range, e.g. `13:00 - 15:50`, `Giờ: 6h30-9h00`
- SUBJECT: subject/course name lines
- ROOM: room lines with a prefix, e.g. `Phòng: B4-502`, `hòng: 1.A302`
- ROOM_CONTINUE: room continuation lines, e.g. `Phòng học Cơ sở 1`
- NOTE: group/teacher/class/major/course-code notes, e.g. `Nhóm: 03`, `GV: ...`
- IGNORE: noise, website, page text, dates, bullets

Files:
- line_train.csv / line_val.csv / line_test.csv: CSV split for scikit-learn
- line_all.csv: all rows combined
- line_*.jsonl: minimal text/label variant
- train_tfidf_line_classifier.py: example training script
- infer_line_labels.py: example inference script

Quick start:
```bash
cd timetable_line_classifier_data
python -m pip install pandas scikit-learn joblib
python train_tfidf_line_classifier.py --train_file line_train.csv --val_file line_val.csv --output timetable_line_classifier.joblib
python infer_line_labels.py --model timetable_line_classifier.joblib --text "Thứ 3
Seminar chuyên để
Nhóm: 03
Phòng: 1.A303-
Phòng học Cơ sở 1
© 13:00 - 15:50"
```

Important: this classifier should support your rule parser, not replace it. Rules should still handle obvious WEEKDAY/TIME/ROOM patterns and final schedule validation.
