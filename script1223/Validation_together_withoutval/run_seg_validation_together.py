"""
Validation_together: Segmentation validation (병렬 실행용)
/home/gpu-agx/zoo/script/Validation_together/run_seg_validation_together.py
- coco2017_val 데이터셋으로 seg 엔진 validation 수행
- start/1 파일 생성 신호를 기다린 뒤 동시에 시작하도록 설계
- CUDA runtime의 cudaDeviceSynchronize() 사용으로 TensorRT 작업 완료까지 정확히 대기
- 결과는 Validation_together 디렉토리에 저장
"""
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from ctypes import cdll, c_int

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO

# CUDA runtime 동기화 설정
try:
    libcudart = cdll.LoadLibrary('libcudart.so')
    cudaDeviceSynchronize = libcudart.cudaDeviceSynchronize
    cudaDeviceSynchronize.restype = c_int
    CUDA_AVAILABLE = True
except (OSError, AttributeError):
    CUDA_AVAILABLE = False
    cudaDeviceSynchronize = None

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "dataset"
RESULT_DIR = ROOT / "result" / "Validation_together_withoutval"
START_FILE = ROOT / "start" / "1"
LOG_PATH = RESULT_DIR / "validation_together.log"
TXT_PATH = RESULT_DIR / "validation_together.txt"
JSON_PATH = RESULT_DIR / "validation_together.json"

COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

SPEC = {
    "name": "yolo11n-seg",
    "engine": "yolo11n-seg.engine",
    "dataset": "coco2017_val",
    "imgsz": 640,
    "task": "segment",
}


def synchronize():
    """CUDA runtime의 device synchronize 사용 (TensorRT 작업 완료까지 대기)"""
    if CUDA_AVAILABLE and cudaDeviceSynchronize:
        result = cudaDeviceSynchronize()
        if result != 0:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
    elif torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def append_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def wait_for_start(path: Path, poll_sec: float = 0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    while not path.exists():
        time.sleep(poll_sec)


def validate_seg() -> Dict[str, Any]:
    """Segmentation validation: coco2017_val"""
    model_path = MODEL_DIR / SPEC["engine"]
    data_path = DATA_DIR / SPEC["dataset"]
    
    log_lines: List[str] = []
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"[{SPEC['name']}] Validation 시작")
    log_lines.append(f"  Model: {model_path}")
    log_lines.append(f"  Dataset: COCO 2017 Val (coco2017_val)")
    log_lines.append(f"  Task: {SPEC['task']}")
    log_lines.append(f"{'='*80}\n")
    
    images_dir = data_path / "val2017"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    image_files = sorted(list(images_dir.glob("*.jpg")) + 
                        list(images_dir.glob("*.png")))[:100]
    
    log_lines.append(f"이미지 파일 수집: {len(image_files)}개 (최대 100개로 제한)")
    log_lines.append("Note: 정확도 계산 없이 추론 시간만 측정합니다.")
    
    model = YOLO(str(model_path))
    
    log_lines.append("Warm-up 실행 중...")
    if image_files:
        _ = model.predict(
            source=str(image_files[0]),
            imgsz=SPEC["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        synchronize()
    log_lines.append("Warm-up 완료\n")
    
    log_lines.append("추론 시간 측정 중...")
    start_time = time.perf_counter()
    
    for img_file in image_files:
        synchronize()  # 이전 작업 완료 대기
        _ = model.predict(
            source=str(img_file),
            imgsz=SPEC["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        synchronize()  # CUDA runtime 동기화로 TensorRT 작업 완료까지 대기
    
    elapsed_time = time.perf_counter() - start_time
    num_images = len(image_files)
    avg_time_per_image = elapsed_time / num_images if num_images > 0 else 0
    
    log_lines.append(f"전체 추론 시간: {elapsed_time:.3f}초")
    log_lines.append(f"이미지 수: {num_images}")
    log_lines.append(f"평균 이미지당 시간: {avg_time_per_image*1000:.3f}ms")
    log_lines.append("")
    
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log_text = "\n".join(log_lines)
    (RESULT_DIR / f"{SPEC['name']}.log").write_text(log_text, encoding="utf-8")
    
    summary = {
        "model": SPEC["name"],
        "task": SPEC["task"],
        "dataset": "coco2017_val",
        "total_time": elapsed_time,
        "num_images": num_images,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "metrics": {},  # 메트릭 없음
    }
    
    summary_text = f"\n{'='*80}\n[{SPEC['name']}]\n"
    summary_text += f"  Dataset: coco2017_val\n"
    summary_text += f"  Task: {SPEC['task']}\n"
    summary_text += f"  전체 시간: {elapsed_time:.3f}초\n"
    summary_text += f"  이미지 수: {num_images}\n"
    summary_text += f"  평균 시간: {avg_time_per_image*1000:.3f}ms\n"
    
    append_text(LOG_PATH, log_text + "\n")
    append_text(TXT_PATH, summary_text + "\n")
    
    json_data = [summary]
    if JSON_PATH.exists():
        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                json_data = existing_data + json_data
        except:
            pass
    
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return summary


def main():
    wait_for_start(START_FILE)
    validate_seg()


if __name__ == "__main__":
    main()

