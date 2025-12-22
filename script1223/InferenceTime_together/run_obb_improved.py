"""
개선된 버전: CUDA runtime 동기화 + tegrastats 로깅
/home/gpu-agx/zoo/script/InferenceTime_together/run_obb_improved.py
- DOTA100 images 100장으로 OBB 엔진 단독 추론 시간 측정
- start/1 파일 생성 신호를 기다린 뒤 동시에 시작하도록 설계
- CUDA runtime의 cudaDeviceSynchronize() 사용으로 TensorRT 작업 완료까지 정확히 대기
"""
import time
import subprocess
import threading
from pathlib import Path
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO

# 공통 동기화 함수 import
import sys
COMMON_SYNC_PATH = Path(__file__).resolve().parent.parent / "common_sync.py"
sys.path.insert(0, str(COMMON_SYNC_PATH.parent))
from common_sync import synchronize

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "dataset"
RESULT_DIR = ROOT / "result" / "InferenceTime_together"
START_FILE = ROOT / "start" / "1"
LOG_PATH = RESULT_DIR / "together_improved.log"
TXT_PATH = RESULT_DIR / "together_improved.txt"
TEGRASTATS_LOG = RESULT_DIR / "tegrastats_obb.log"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPEC = {
    "name": "yolo11n-obb",
    "engine": "yolo11n-obb.engine",
    "dataset": "DOTA100/images",
    "imgsz": 640,
    "limit": 100,
}


class TegrastatsMonitor:
    """tegrastats를 백그라운드로 실행하고 로깅"""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.process: Optional[subprocess.Popen] = None
        self.running = False

    def start(self):
        if self.running:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.process = subprocess.Popen(
                ['tegrastats', '--interval', '100'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.running = True
            threading.Thread(target=self._log_worker, daemon=True).start()
        except FileNotFoundError:
            print("Warning: tegrastats not found, skipping GPU monitoring")

    def _log_worker(self):
        if not self.process:
            return
        with self.log_path.open('a', encoding='utf-8') as f:
            f.write(f"=== Tegrastats started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            for line in self.process.stdout:
                f.write(line)
                f.flush()

    def stop(self):
        if self.process and self.running:
            self.running = False
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            if self.log_path.exists():
                with self.log_path.open('a', encoding='utf-8') as f:
                    f.write(f"=== Tegrastats stopped at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")


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
        f"[{SPEC['name']}] dataset={data_path} count={len(images)} imgsz={SPEC['imgsz']}",
        "Using CUDA runtime cudaDeviceSynchronize() for accurate TensorRT synchronization"
    ]
    for img in images:
        synchronize()  # 이전 작업 완료 대기
        t0 = time.perf_counter()
        _ = model.predict(
            source=str(img),
            imgsz=SPEC["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        synchronize()  # CUDA runtime 동기화로 TensorRT 작업 완료까지 대기
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        log_lines.append(f"{img}: {elapsed*1000:.3f} ms")

    avg = sum(times) / len(times)
    log_lines.append(f"avg: {avg*1000:.3f} ms\n")
    log_text = "\n".join(log_lines)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"{SPEC['name']}_improved.log").write_text(log_text)
    summary = f"{SPEC['name']}: {avg*1000:.3f} ms (n={len(times)})"
    (RESULT_DIR / f"{SPEC['name']}_improved.txt").write_text(summary)

    append_text(LOG_PATH, log_text + "\n")
    append_text(TXT_PATH, summary + "\n")
    return summary


def main():
    # tegrastats 모니터링 시작
    monitor = TegrastatsMonitor(TEGRASTATS_LOG)
    monitor.start()

    try:
        wait_for_start(START_FILE)
        measure_model()
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()


