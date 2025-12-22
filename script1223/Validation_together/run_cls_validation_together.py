"""
Validation_together: Classification validation (병렬 실행용)
/home/gpu-agx/zoo/script/Validation_together/run_cls_validation_together.py
- ImageNet1k_100 데이터셋으로 cls 엔진 validation 수행
- start/1 파일 생성 신호를 기다린 뒤 동시에 시작하도록 설계
- CUDA runtime의 cudaDeviceSynchronize() 사용으로 TensorRT 작업 완료까지 정확히 대기
- 결과는 Validation_together 디렉토리에 저장
"""
import time
import json
import csv
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
    "name": "yolo11n-cls",
    "engine": "yolo11n-cls.engine",
    "dataset": "ImageNet1k_100",
    "imgsz": 224,
    "task": "classify",
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


def validate_classify() -> Dict[str, Any]:
    """Classification validation: ImageNet1k_100"""
    model_path = MODEL_DIR / SPEC["engine"]
    data_path = DATA_DIR / SPEC["dataset"]
    
    log_lines: List[str] = []
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"[{SPEC['name']}] Validation 시작")
    log_lines.append(f"  Model: {model_path}")
    log_lines.append(f"  Dataset: {data_path}")
    log_lines.append(f"  Note: ImageNet 1000개 클래스 validation 100개 이미지")
    log_lines.append(f"{'='*80}\n")
    
    val_dir = data_path / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    val_images = sorted([p for p in val_dir.rglob("*.jpg")] + 
                       [p for p in val_dir.rglob("*.png")])
    if not val_images:
        raise RuntimeError(f"No images found in {val_dir}")
    
    log_lines.append(f"검증 이미지: {len(val_images)}개")
    
    csv_file = data_path / "val100.csv"
    gt_dict = {}
    
    if csv_file.exists():
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = row['image_path']
                class_id = int(row['class_id'])
                full_path = (data_path / image_path).resolve()
                gt_dict[str(full_path)] = class_id
        log_lines.append(f"Ground truth 로드 완료: {len(gt_dict)}개 (CSV 파일 사용)")
    else:
        json_file = data_path / "annotations_val100.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
            image_id_to_file = {img['id']: img['file_name'] for img in ann_data['images']}
            for ann in ann_data['annotations']:
                image_id = ann['image_id']
                category_id = ann['category_id']
                if image_id in image_id_to_file:
                    file_name = image_id_to_file[image_id]
                    full_path = (data_path / file_name).resolve()
                    gt_dict[str(full_path)] = category_id
            log_lines.append(f"Ground truth 로드 완료: {len(gt_dict)}개 (JSON 파일 사용)")
        else:
            log_lines.append("⚠️  Annotation 파일을 찾을 수 없습니다. 폴더명에서 클래스 ID 추출합니다.")
            for img_path in val_images:
                class_folder = img_path.parent.name
                if class_folder.isdigit():
                    gt_dict[str(img_path.resolve())] = int(class_folder)
    
    log_lines.append(f"Ground truth 매핑: {len(gt_dict)}개")
    
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
    
    log_lines.append("추론 시간 측정 및 정확도 계산 중...")
    start_time = time.perf_counter()
    
    results_list = []
    correct_top1 = 0
    correct_top5 = 0
    total_with_gt = 0
    
    for img_path in val_images:
        img_str = str(img_path.resolve())
        result = model.predict(
            source=img_str,
            imgsz=SPEC["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        results_list.append(result[0])
        
        if img_str in gt_dict and hasattr(result[0], 'probs') and result[0].probs is not None:
            gt_class_id = gt_dict[img_str]
            total_with_gt += 1
            predicted_top1 = int(result[0].probs.top1)
            
            if predicted_top1 == gt_class_id:
                correct_top1 += 1
            
            if hasattr(result[0].probs, 'top5') and result[0].probs.top5 is not None:
                top5_list = result[0].probs.top5
                if torch is not None and torch.is_tensor(top5_list):
                    top5_list = top5_list.cpu().numpy().tolist()
                elif hasattr(top5_list, 'tolist'):
                    top5_list = top5_list.tolist()
                else:
                    top5_list = list(top5_list)
                
                if gt_class_id in top5_list[:5]:
                    correct_top5 += 1
    
    synchronize()
    elapsed_time = time.perf_counter() - start_time
    num_images = len(val_images)
    avg_time_per_image = elapsed_time / num_images if num_images > 0 else 0
    
    top1_accuracy = correct_top1 / total_with_gt if total_with_gt > 0 else 0.0
    top5_accuracy = correct_top5 / total_with_gt if total_with_gt > 0 else 0.0
    
    metrics = {
        "top1": top1_accuracy,
        "top5": top5_accuracy,
        "official_top1": top1_accuracy,
        "official_top5": top5_accuracy,
    }
    
    log_lines.append(f"전체 추론 시간: {elapsed_time:.3f}초")
    log_lines.append(f"이미지 수: {num_images}")
    log_lines.append(f"평균 이미지당 시간: {avg_time_per_image*1000:.3f}ms")
    if total_with_gt > 0:
        log_lines.append(f"정확도 계산 대상: {total_with_gt}개")
        log_lines.append(f"Top-1 정확도: {correct_top1}/{total_with_gt} = {top1_accuracy:.4f}")
        log_lines.append(f"Top-5 정확도: {correct_top5}/{total_with_gt} = {top5_accuracy:.4f}")
        log_lines.append(f"[공식 지표] Top-1: {top1_accuracy:.4f}")
        log_lines.append(f"[공식 지표] Top-5: {top5_accuracy:.4f}")
    log_lines.append("")
    
    # 결과 저장
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log_text = "\n".join(log_lines)
    (RESULT_DIR / f"{SPEC['name']}.log").write_text(log_text, encoding="utf-8")
    
    summary = {
        "model": SPEC["name"],
        "task": SPEC["task"],
        "dataset": SPEC["dataset"],
        "total_time": elapsed_time,
        "num_images": num_images,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "metrics": metrics,
    }
    
    summary_text = f"\n{'='*80}\n[{SPEC['name']}]\n"
    summary_text += f"  Dataset: {SPEC['dataset']}\n"
    summary_text += f"  Task: {SPEC['task']}\n"
    summary_text += f"  전체 시간: {elapsed_time:.3f}초\n"
    summary_text += f"  이미지 수: {num_images}\n"
    summary_text += f"  평균 시간: {avg_time_per_image*1000:.3f}ms\n"
    summary_text += "  메트릭:\n"
    summary_text += f"    Top-1 Accuracy: {top1_accuracy:.4f}\n"
    summary_text += f"    Top-5 Accuracy: {top5_accuracy:.4f}\n"
    summary_text += f"    [공식 지표] Top-1: {top1_accuracy:.4f}\n"
    summary_text += f"    [공식 지표] Top-5: {top5_accuracy:.4f}\n"
    
    append_text(LOG_PATH, log_text + "\n")
    append_text(TXT_PATH, summary_text + "\n")
    
    # JSON 저장
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
    validate_classify()


if __name__ == "__main__":
    main()


