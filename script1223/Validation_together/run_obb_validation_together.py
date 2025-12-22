"""
Validation_together: OBB validation (병렬 실행용)
/home/gpu-agx/zoo/script/Validation_together/run_obb_validation_together.py
- DOTAv1.5-1024_val 데이터셋으로 obb 엔진 validation 수행
- start/1 파일 생성 신호를 기다린 뒤 동시에 시작하도록 설계
- CUDA runtime의 cudaDeviceSynchronize() 사용으로 TensorRT 작업 완료까지 정확히 대기
- 결과는 Validation_together 디렉토리에 저장
"""
import time
import json
import yaml
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
RESULT_DIR = ROOT / "result" / "Validation_together"
START_FILE = ROOT / "start" / "1"
LOG_PATH = RESULT_DIR / "validation_together.log"
TXT_PATH = RESULT_DIR / "validation_together.txt"
JSON_PATH = RESULT_DIR / "validation_together.json"

SPEC = {
    "name": "yolo11n-obb",
    "engine": "yolo11n-obb.engine",
    "dataset": "DOTAv1.5-1024_val",
    "imgsz": 640,
    "task": "obb",
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


def validate_obb() -> Dict[str, Any]:
    """OBB validation: DOTAv1.5-1024_val"""
    model_path = MODEL_DIR / SPEC["engine"]
    data_path = DATA_DIR / SPEC["dataset"]
    
    val_images_dir = data_path / "images" / "val"
    val_labels_dir = data_path / "labels" / "val"
    
    log_lines: List[str] = []
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"[{SPEC['name']}] Validation 시작")
    log_lines.append(f"  Model: {model_path}")
    log_lines.append(f"  Dataset: DOTA v1.5-1024 Val")
    log_lines.append(f"  Images: {val_images_dir}")
    log_lines.append(f"{'='*80}\n")
    
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {val_images_dir}")
    
    val_images = sorted([f for f in val_images_dir.iterdir()
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])[:100]

    train_images_dir = data_path / "images" / "train"
    train_images = []
    if train_images_dir.exists():
        train_images = sorted([f for f in train_images_dir.iterdir()
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    combined_images = val_images
    if len(combined_images) < 100 and train_images:
        needed = 100 - len(combined_images)
        combined_images = combined_images + train_images[:needed]
    
    combined_images = combined_images[:100]

    if not combined_images:
        raise RuntimeError(f"No images found in {val_images_dir}")

    log_lines.append(
        f"검증 이미지: {len(combined_images)}개 (최대 100개로 제한) "
        f"(val: {min(len(val_images), 100)}개, train 사용: {max(0, len(combined_images)-len(val_images))}개)"
    )
    
    ground_truth_obbs = []
    if val_labels_dir.exists():
        for label_file in val_labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                ground_truth_obbs.extend(lines)
    
    train_labels_dir = data_path / "labels" / "train"
    if train_labels_dir.exists() and len(combined_images) > len(val_images):
        for label_file in train_labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                ground_truth_obbs.extend(lines)
    
    log_lines.append(f"Ground-truth OBB: {len(ground_truth_obbs)}개")
    
    model = YOLO(str(model_path))
    
    log_lines.append("Warm-up 실행 중...")
    _ = model.predict(
        source=str(val_images[0]),
        imgsz=SPEC["imgsz"],
        device=0,
        save=False,
        verbose=False,
    )
    synchronize()
    log_lines.append("Warm-up 완료\n")
    
    log_lines.append("추론 시간 측정 중...")
    start_time = time.perf_counter()
    
    results_list = []
    total_detected_obbs = 0
    
    for img_path in combined_images:
        result = model.predict(
            source=str(img_path),
            imgsz=SPEC["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        results_list.append(result[0])
        
        if result[0].obb is not None:
            num_detections = len(result[0].obb)
            total_detected_obbs += num_detections
    
    synchronize()
    elapsed_time = time.perf_counter() - start_time
    
    total_processed = len(results_list)
    avg_time_per_image = elapsed_time / total_processed if total_processed > 0 else 0
    
    official_oriented_map = None
    try:
        log_lines.append("공식 지표 계산 중 (YOLO val() 메서드)...")
        dataset_yaml = data_path / "dota1.5_yolo.yaml"
        if not dataset_yaml.exists():
            dataset_yaml = data_path / "DOTA1.5.yaml"
        if not dataset_yaml.exists():
            dataset_yaml = data_path / "dota.yaml"
        if not dataset_yaml.exists():
            dataset_yaml = None
        
        if dataset_yaml and dataset_yaml.exists():
            with open(dataset_yaml, "r", encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f)
            if isinstance(data_cfg, dict):
                data_cfg["path"] = str(data_path)
                val_cache = data_path / "labels" / "val.cache"
                if val_cache.exists():
                    val_cache.unlink()
                temp_yaml = data_path / "temp_dota1.5_val.yaml"
                with open(temp_yaml, "w", encoding="utf-8") as f:
                    yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True)
                try:
                    val_results = model.val(
                        data=str(temp_yaml),
                        imgsz=SPEC["imgsz"],
                        device=0,
                        verbose=False,
                        plots=False,
                    )
                except Exception as val_error:
                    log_lines.append(f"⚠️  val() 실행 중 에러: {str(val_error)}")
                    val_results = None
                finally:
                    if temp_yaml.exists():
                        temp_yaml.unlink()
            else:
                val_results = None
            
            if val_results is not None:
                try:
                    if hasattr(val_results, "obb"):
                        obb_metrics = val_results.obb
                        if hasattr(obb_metrics, "map50_95"):
                            official_oriented_map = float(obb_metrics.map50_95)
                        elif hasattr(obb_metrics, "map"):
                            official_oriented_map = float(obb_metrics.map)
                    elif hasattr(val_results, "map50_95"):
                        official_oriented_map = float(val_results.map50_95)
                    elif hasattr(val_results, "map"):
                        official_oriented_map = float(val_results.map)
                    
                    if official_oriented_map is not None:
                        log_lines.append(f"공식 지표 계산 완료: {official_oriented_map:.4f}")
                except Exception as extract_error:
                    log_lines.append(f"⚠️  지표 추출 중 에러: {str(extract_error)}")
        else:
            log_lines.append("⚠️  데이터셋 yaml 파일을 찾을 수 없어 공식 지표 계산을 건너뜁니다.")
    except Exception as e:
        log_lines.append(f"⚠️  공식 지표 계산 실패: {str(e)}")
    
    metrics = {
        "total_detections": total_detected_obbs,
        "avg_detections_per_image": total_detected_obbs / total_processed if total_processed > 0 else 0,
    }
    if official_oriented_map is not None:
        metrics["official_oriented_map50_95"] = official_oriented_map
    
    log_lines.append(f"전체 추론 시간: {elapsed_time:.3f}초")
    log_lines.append(f"이미지 수: {total_processed}")
    log_lines.append(f"평균 이미지당 시간: {avg_time_per_image*1000:.3f}ms")
    log_lines.append(f"탐지된 OBB: {total_detected_obbs}개")
    log_lines.append(f"이미지당 평균: {metrics['avg_detections_per_image']:.2f}개")
    if official_oriented_map is not None:
        log_lines.append(f"[공식 지표] oriented mAP@[0.5:0.95]: {official_oriented_map:.4f}")
    log_lines.append("")
    
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log_text = "\n".join(log_lines)
    (RESULT_DIR / f"{SPEC['name']}.log").write_text(log_text, encoding="utf-8")
    
    summary = {
        "model": SPEC["name"],
        "task": SPEC["task"],
        "dataset": "DOTAv1.5-1024_val",
        "total_time": elapsed_time,
        "num_images": total_processed,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "metrics": metrics,
    }
    
    summary_text = f"\n{'='*80}\n[{SPEC['name']}]\n"
    summary_text += f"  Dataset: DOTAv1.5-1024_val\n"
    summary_text += f"  Task: {SPEC['task']}\n"
    summary_text += f"  전체 시간: {elapsed_time:.3f}초\n"
    summary_text += f"  이미지 수: {total_processed}\n"
    summary_text += f"  평균 시간: {avg_time_per_image*1000:.3f}ms\n"
    summary_text += "  메트릭:\n"
    summary_text += f"    Total Detections: {total_detected_obbs}\n"
    summary_text += f"    Avg Detections/Image: {metrics['avg_detections_per_image']:.2f}\n"
    if official_oriented_map is not None:
        summary_text += f"    [공식 지표] oriented mAP@[0.5:0.95]: {official_oriented_map:.4f}\n"
    
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
    validate_obb()


if __name__ == "__main__":
    main()


