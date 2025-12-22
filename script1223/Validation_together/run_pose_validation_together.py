"""
Validation_together: Pose estimation validation (병렬 실행용)
/home/gpu-agx/zoo/script/Validation_together/run_pose_validation_together.py
- coco2017_val 데이터셋으로 pose 엔진 validation 수행
- start/1 파일 생성 신호를 기다린 뒤 동시에 시작하도록 설계
- CUDA runtime의 cudaDeviceSynchronize() 사용으로 TensorRT 작업 완료까지 정확히 대기
- 결과는 Validation_together 디렉토리에 저장
"""
import time
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
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
    "name": "yolo11n-pose",
    "engine": "yolo11n-pose.engine",
    "dataset": "coco2017_val",
    "imgsz": 640,
    "task": "pose",
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


def validate_pose() -> Dict[str, Any]:
    """Pose estimation validation: coco2017_val"""
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
    
    gt_by_image_path = defaultdict(list)
    total_gt_objects = 0
    
    annotation_file = data_path / "annotations" / "person_keypoints_val2017.json"
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in image_id_to_file:
            file_name = image_id_to_file[image_id]
            img_path = images_dir / file_name
            img_path_str = str(img_path.resolve())
            
            bbox = ann['bbox']
            category_id = ann['category_id']
            
            gt_obj = {
                "category": category_id,
                "bbox": bbox,
                "coco_format": True
            }
            
            if "keypoints" in ann:
                gt_obj["keypoints"] = ann["keypoints"]
                gt_obj["num_keypoints"] = ann.get("num_keypoints", 0)
            
            gt_by_image_path[img_path_str].append(gt_obj)
            total_gt_objects += 1
    
    log_lines.append(f"이미지 파일: {len(image_files)}개")
    log_lines.append(f"Ground-truth 객체: {total_gt_objects}개")
    log_lines.append(f"Ground-truth가 있는 이미지: {len(gt_by_image_path)}개")
    
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
    
    log_lines.append("추론 시간 측정 및 정확도 계산 중...")
    start_time = time.perf_counter()
    
    results_list = []
    total_detected = 0
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    for img_file in image_files:
        img_path_str = str(img_file.resolve())
        
        result = model.predict(
            source=img_path_str,
            imgsz=SPEC["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        results_list.append(result[0])
        
        gt_objects = gt_by_image_path.get(img_path_str, [])
        
        if result[0].boxes is not None:
            detected_count = len(result[0].boxes)
            total_detected += detected_count
            
            gt_count = len(gt_objects)
            matched = min(detected_count, gt_count)
            tp_count += matched
            
            if detected_count > gt_count:
                fp_count += (detected_count - gt_count)
            
            if gt_count > detected_count:
                fn_count += (gt_count - detected_count)
    
    synchronize()
    elapsed_time = time.perf_counter() - start_time
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1 = 2 * tp_count / (2 * tp_count + fp_count + fn_count) if (2 * tp_count + fp_count + fn_count) > 0 else 0.0
    
    official_map = None
    try:
        log_lines.append("공식 지표 계산 중 (YOLO val() 메서드)...")
        coco_root = data_path
        temp_yaml = coco_root / "temp_coco2017_val.yaml"
        annotation_file = coco_root / "annotations" / "person_keypoints_val2017.json"
        
        data_cfg: Dict[str, Any] = {
            "path": str(coco_root),
            "train": "val2017",
            "val": "val2017",
            "names": COCO80_NAMES,
            "nc": len(COCO80_NAMES),
            "kpt_shape": [17, 3],
        }
        
        with open(temp_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True)
        
        try:
            val_results = model.val(
                data=str(temp_yaml),
                imgsz=SPEC["imgsz"],
                device=0,
                verbose=False,
                save_json=False,
            )
        except Exception as val_error:
            log_lines.append(f"⚠️  val() 메서드 실행 중 에러: {str(val_error)}")
            val_results = None
        finally:
            if temp_yaml.exists():
                temp_yaml.unlink()
        
        if val_results is not None:
            try:
                if hasattr(val_results, "pose"):
                    pose_metrics = val_results.pose
                    if hasattr(pose_metrics, "map50_95"):
                        official_map = float(pose_metrics.map50_95)
                    elif hasattr(pose_metrics, "map"):
                        official_map = float(pose_metrics.map)
                elif hasattr(val_results, "map50_95"):
                    official_map = float(val_results.map50_95)
                elif hasattr(val_results, "map"):
                    official_map = float(val_results.map)
                
                if official_map is not None:
                    log_lines.append(f"공식 지표 계산 완료: {official_map:.4f}")
            except Exception as extract_error:
                log_lines.append(f"⚠️  지표 추출 중 에러: {str(extract_error)}")
    except Exception as e:
        log_lines.append(f"⚠️  공식 지표 계산 실패: {str(e)}")
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if official_map is not None:
        metrics["official_oks_map50_95"] = official_map
    
    num_images = len(results_list)
    avg_time_per_image = elapsed_time / num_images if num_images > 0 else 0
    
    log_lines.append(f"전체 추론 시간: {elapsed_time:.3f}초")
    log_lines.append(f"이미지 수: {num_images}")
    log_lines.append(f"평균 이미지당 시간: {avg_time_per_image*1000:.3f}ms")
    log_lines.append(f"탐지된 객체: {total_detected}개")
    log_lines.append(f"True Positive: {tp_count}개")
    log_lines.append(f"False Positive: {fp_count}개")
    log_lines.append(f"False Negative: {fn_count}개")
    if official_map is not None:
        log_lines.append(f"[공식 지표] OKS mAP@[0.5:0.95]: {official_map:.4f}")
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
        "metrics": metrics,
    }
    
    summary_text = f"\n{'='*80}\n[{SPEC['name']}]\n"
    summary_text += f"  Dataset: coco2017_val\n"
    summary_text += f"  Task: {SPEC['task']}\n"
    summary_text += f"  전체 시간: {elapsed_time:.3f}초\n"
    summary_text += f"  이미지 수: {num_images}\n"
    summary_text += f"  평균 시간: {avg_time_per_image*1000:.3f}ms\n"
    summary_text += "  메트릭:\n"
    summary_text += f"    Precision: {precision:.4f}\n"
    summary_text += f"    Recall: {recall:.4f}\n"
    summary_text += f"    F1-Score: {f1:.4f}\n"
    if official_map is not None:
        summary_text += f"    [공식 지표] OKS mAP@[0.5:0.95]: {official_map:.4f}\n"
    
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
    validate_pose()


if __name__ == "__main__":
    main()


