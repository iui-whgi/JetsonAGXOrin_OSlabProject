"""
/home/gpu-agx/zoo/script/InferenceTime_together/run_pose.py
- coco128 train2017 100장으로 pose 엔진 단독 추론 시간 측정
- start/1 파일 생성 신호를 기다린 뒤 동시에 시작하도록 설계
"""
import time
from pathlib import Path
from typing import List

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "dataset"
RESULT_DIR = ROOT / "result" / "InferenceTime_together"
START_FILE = ROOT / "start" / "1"
LOG_PATH = RESULT_DIR / "together.log"
TXT_PATH = RESULT_DIR / "together.txt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPEC = {
    "name": "yolo11n-pose",
    "engine": "yolo11n-pose.engine",
    "dataset": "coco128/images/train2017",
    "imgsz": 640,
    "limit": 100,
}


def append_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def wait_for_start(path: Path, poll_sec: float = 0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    while not path.exists():
        time.sleep(poll_sec)


def collect_images(dataset_root: Path, limit: int) -> List[Path]:
    images = sorted(
        p for p in dataset_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    return images[:limit] if limit else images


def synchronize():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_model() -> str:
    model_path = MODEL_DIR / SPEC["engine"]
    data_path = DATA_DIR / SPEC["dataset"]

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"dataset not found: {data_path}")

    images = collect_images(data_path, SPEC["limit"])
    if not images:
        raise RuntimeError(f"no images found under {data_path}")

    model = YOLO(str(model_path))

    # Warm-up once to avoid first-run overhead.
    _ = model.predict(
        source=str(images[0]),
        imgsz=SPEC["imgsz"],
        device=0,
        save=False,
        verbose=False,
    )
    synchronize()

    times = []
    log_lines = [
        f"[{SPEC['name']}] dataset={data_path} count={len(images)} imgsz={SPEC['imgsz']}"
    ]
    for img in images:
        t0 = time.perf_counter()
        _ = model.predict(
            source=str(img),
            imgsz=SPEC["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        log_lines.append(f"{img}: {elapsed*1000:.3f} ms")

    avg = sum(times) / len(times)
    log_lines.append(f"avg: {avg*1000:.3f} ms\n")
    log_text = "\n".join(log_lines)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"{SPEC['name']}.log").write_text(log_text)
    summary = f"{SPEC['name']}: {avg*1000:.3f} ms (n={len(times)})"
    (RESULT_DIR / f"{SPEC['name']}.txt").write_text(summary)

    append_text(LOG_PATH, log_text + "\n")
    append_text(TXT_PATH, summary + "\n")
    return summary


def main():
    wait_for_start(START_FILE)
    measure_model()


if __name__ == "__main__":
    main()


