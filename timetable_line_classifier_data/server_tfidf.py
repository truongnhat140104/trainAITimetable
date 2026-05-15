from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

app = FastAPI()

model = joblib.load("timetable_line_classifier.joblib")


class ParseRequest(BaseModel):
    text: str


def normalize_text(text: str) -> str:
    return (
        text.replace("©", "")
            .replace("–", "-")
            .replace("—", "-")
            .strip()
    )


def weekday_from_line(line: str):
    lower = line.lower()

    if "thứ 2" in lower or "thứ hai" in lower:
        return 2
    if "thứ 3" in lower or "thứ ba" in lower:
        return 3
    if "thứ 4" in lower or "thứ tư" in lower:
        return 4
    if "thứ 5" in lower or "thứ năm" in lower:
        return 5
    if "thứ 6" in lower or "thứ sáu" in lower:
        return 6
    if "thứ 7" in lower or "thứ bảy" in lower:
        return 7
    if "chủ nhật" in lower:
        return 1

    return None


def parse_time_range(line: str):
    cleaned = (
        line.lower()
            .replace("giờ:", "")
            .replace("©", "")
            .replace(" ", "")
            .replace("h", ":")
            .replace("g", ":")
            .replace("->", "-")
    )

    match = re.search(r"(\d{1,2}:\d{2})-(\d{1,2}:\d{2})", cleaned)

    if not match:
        return None

    start = match.group(1)
    end = match.group(2)

    if len(start) == 4:
        start = "0" + start

    if len(end) == 4:
        end = "0" + end

    return start, end


def classify_line(line: str):
    lower = line.lower().strip()

    if weekday_from_line(line) is not None:
        return "WEEKDAY"

    if parse_time_range(line) is not None:
        return "TIME"

    if lower.startswith("phòng:") or lower.startswith("hòng:"):
        return "ROOM"

    if "phòng học" in lower:
        return "ROOM_CONTINUE"

    if (
        lower.startswith("gv:")
        or lower.startswith("nhóm:")
        or lower.startswith("khóm:")
        or lower.startswith("lớp:")
        or lower.startswith("buổi:")
        or "ngành" in lower
    ):
        return "NOTE"

    if (
        "@" in lower
        or "http" in lower
        or "www" in lower
        or "ndaotao" in lower
        or lower == "trang"
        or re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", lower)
    ):
        return "IGNORE"

    return model.predict([line])[0]


def clean_subject(text: str):
    return (
        text.replace("chuyên để", "chuyên đề")
            .replace("trinh", "trình")
            .replace("Đổ án", "Đồ án")
            .replace("Kinh tẽ", "Kinh tế")
            .strip()
    )


def clean_room(lines):
    room_parts = []

    for line in lines:
        cleaned = (
            line.replace("Phòng:", "")
                .replace("phòng:", "")
                .replace("hòng:", "")
                .strip()
        )
        room_parts.append(cleaned)

    return " ".join(room_parts).strip()


def flush_item(items, current_weekday, buffer, time_range):
    if current_weekday is None or time_range is None:
        return

    subject_lines = []
    room_lines = []
    note_lines = []

    for line, label in buffer:
        if label == "SUBJECT":
            subject_lines.append(line)
        elif label in ["ROOM", "ROOM_CONTINUE"]:
            room_lines.append(line)
        elif label == "NOTE":
            note_lines.append(line)

    subject = clean_subject(" ".join(subject_lines))

    if not subject:
        return

    start_time, end_time = time_range

    items.append({
        "subjectName": subject,
        "weekday": current_weekday,
        "startTime": start_time,
        "endTime": end_time,
        "room": clean_room(room_lines) if room_lines else None,
        "note": ", ".join(note_lines) if note_lines else None
    })


@app.post("/parse-timetable")
def parse_timetable(request: ParseRequest):
    lines = [
        normalize_text(line)
        for line in request.text.splitlines()
        if normalize_text(line)
    ]

    items = []
    current_weekday = None
    buffer = []

    for line in lines:
        label = classify_line(line)

        if label == "WEEKDAY":
            current_weekday = weekday_from_line(line)
            buffer = []
            continue

        if label == "IGNORE":
            continue

        if label == "TIME":
            time_range = parse_time_range(line)
            flush_item(items, current_weekday, buffer, time_range)
            buffer = []
            continue

        buffer.append((line, label))

    return {
        "items": items
    }