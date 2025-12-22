'''
/home/gpu-agx/zoo/script/InferenceTime_alone/alone_sequence.py에 5개 엔진(클래스/디텍션/포즈/세그/OBB)을 순차 실행해 이미지별 추론 시간과 평균을 측정하고 zoo/result/InferenceTime_alone/alone_sequence.log,
alone_sequence.txt에 기록하도록 추가했습니다. 데이터셋은 CLS→CIFAR100 100장, detect/pose/seg→coco128/images/train2017 100장, OBB→DOTA100/images 100장 사용하며, 첫 장은 
웜업으로 측정에서 제외합니다. GPU 동기화(torch 유무 체크)로 순수 추론 시간만 집계합니다.
'''
import time
from pathlib import Path
from typing import List, Dict

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "dataset"
RESULT_DIR = ROOT / "result" / "InferenceTime_alone"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_SPECS: List[Dict] = [
    {
        "name": "yolo11n-cls",
        "engine": "yolo11n-cls.engine",
        "dataset": "CIFAR100",
        "imgsz": 224,
        "limit": 100,
    },
    {
        "name": "yolo11n-detect",
        "engine": "yolo11n-detect.engine",
        "dataset": "coco128/images/train2017",
        "imgsz": 640,
        "limit": 100,
    },
    {
        "name": "yolo11n-pose",
        "engine": "yolo11n-pose.engine",
        "dataset": "coco128/images/train2017",
        "imgsz": 640,
        "limit": 100,
    },
    {
        "name": "yolo11n-seg",
        "engine": "yolo11n-seg.engine",
        "dataset": "coco128/images/train2017",
        "imgsz": 640,
        "limit": 100,
    },
    {
        "name": "yolo11n-obb",
        "engine": "yolo11n-obb.engine",
        "dataset": "DOTA100/images",
        "imgsz": 640,
        "limit": 100,
    },
]


def collect_images(dataset_root: Path, limit: int) -> List[Path]:
    images = sorted(
        p for p in dataset_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    return images[:limit] if limit else images


def synchronize():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_model(spec: Dict, log_lines: List[str]) -> str:
    model_path = MODEL_DIR / spec["engine"]
    data_path = DATA_DIR / spec["dataset"]

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"dataset not found: {data_path}")

    images = collect_images(data_path, spec["limit"])
    if not images:
        raise RuntimeError(f"no images found under {data_path}")

    model = YOLO(str(model_path))

    # Warm-up once to avoid first-run overhead in measurements.
    _ = model.predict(
        source=str(images[0]),
        imgsz=spec["imgsz"],
        device=0,
        save=False,
        verbose=False,
    )
    synchronize()

    times = []
    log_lines.append(
        f"[{spec['name']}] dataset={data_path} count={len(images)} imgsz={spec['imgsz']}"
    )
    for img in images:
        t0 = time.perf_counter()
        _ = model.predict(
            source=str(img),
            imgsz=spec["imgsz"],
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
    return f"{spec['name']}: {avg*1000:.3f} ms (n={len(times)})"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULT_DIR / "alone_sequence.log"
    txt_path = RESULT_DIR / "alone_sequence.txt"

    log_lines: List[str] = []
    summary: List[str] = []

    for spec in MODEL_SPECS:
        summary_line = measure_model(spec, log_lines)
        summary.append(summary_line)

    log_path.write_text("\n".join(log_lines))
    txt_path.write_text("\n".join(summary))


if __name__ == "__main__":
    main()
