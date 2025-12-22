'''
/home/gpu-agx/zoo/script/Validation_alone/validation_alone.py
각 데이터셋에 대해 추론을 수행하고 추론 시간과 정확도(accuracy, precision, recall, F1-score)를 측정하여
zoo/result/Validation_alone/에 log와 txt 파일로 저장합니다.
각 모델 타입별로 적절한 validation 데이터셋을 사용합니다.

예전 pod validation 스크립트 방식을 참고하여 predict()를 사용하고 ground truth와 직접 비교합니다.

중요: 각 engine별로 순차적으로 실행됩니다. 동시/병렬 실행은 절대 수행하지 않습니다.
'''
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from collections import defaultdict

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO
import yaml


ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "dataset"
RESULT_DIR = ROOT / "result" / "Validation_alone_withoutval"

# coco128은 YOLO 형식 labels를 사용 (txt 파일)

MODEL_SPECS: List[Dict] = [
    {
        "name": "yolo11n-cls",
        "engine": "yolo11n-cls.engine",
        "dataset": "ImageNet1k_100",  # ImageNet 1000개 클래스 validation 100개 이미지
        "imgsz": 224,
        "task": "classify",
    },
    {
        "name": "yolo11n-detect",
        "engine": "yolo11n-detect.engine",
        "dataset": "coco2017_val",
        "imgsz": 640,
        "task": "detect",
    },
    {
        "name": "yolo11n-pose",
        "engine": "yolo11n-pose.engine",
        "dataset": "coco2017_val",
        "imgsz": 640,
        "task": "pose",
    },
    {
        "name": "yolo11n-seg",
        "engine": "yolo11n-seg.engine",
        "dataset": "coco2017_val",
        "imgsz": 640,
        "task": "segment",
    },
    {
        "name": "yolo11n-obb",
        "engine": "yolo11n-obb.engine",
        "dataset": "DOTAv1.5-1024_val",
        "imgsz": 640,
        "task": "obb",
    },
]


# COCO 80 classes (공식 클래스 이름)
COCO80_NAMES: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def synchronize():
    """GPU 동기화"""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def format_metrics(metrics: Dict[str, Any], task: str) -> str:
    """메트릭을 포맷팅하여 문자열로 반환"""
    lines = []
    
    if task == "classify":
        if "top1" in metrics:
            lines.append(f"  Top-1 Accuracy: {metrics['top1']:.4f}")
        if "top5" in metrics:
            lines.append(f"  Top-5 Accuracy: {metrics['top5']:.4f}")
        # 공식 지표
        if "official_top1" in metrics:
            lines.append(f"  [공식 지표] Top-1: {metrics['official_top1']:.4f}")
        if "official_top5" in metrics:
            lines.append(f"  [공식 지표] Top-5: {metrics['official_top5']:.4f}")
    elif task == "detect":
        if "precision" in metrics:
            lines.append(f"  Precision: {metrics['precision']:.4f}")
        if "recall" in metrics:
            lines.append(f"  Recall: {metrics['recall']:.4f}")
        if "f1" in metrics:
            lines.append(f"  F1-Score: {metrics['f1']:.4f}")
        # 공식 지표
        if "official_map50_95" in metrics:
            lines.append(f"  [공식 지표] mAP@[0.5:0.95]: {metrics['official_map50_95']:.4f}")
    elif task == "segment":
        if "precision" in metrics:
            lines.append(f"  Precision: {metrics['precision']:.4f}")
        if "recall" in metrics:
            lines.append(f"  Recall: {metrics['recall']:.4f}")
        if "f1" in metrics:
            lines.append(f"  F1-Score: {metrics['f1']:.4f}")
        # 공식 지표
        if "official_mask_map50_95" in metrics:
            lines.append(f"  [공식 지표] mask mAP@[0.5:0.95]: {metrics['official_mask_map50_95']:.4f}")
    elif task == "pose":
        if "precision" in metrics:
            lines.append(f"  Precision: {metrics['precision']:.4f}")
        if "recall" in metrics:
            lines.append(f"  Recall: {metrics['recall']:.4f}")
        if "f1" in metrics:
            lines.append(f"  F1-Score: {metrics['f1']:.4f}")
        # 공식 지표
        if "official_oks_map50_95" in metrics:
            lines.append(f"  [공식 지표] OKS mAP@[0.5:0.95]: {metrics['official_oks_map50_95']:.4f}")
    elif task == "obb":
        if "total_detections" in metrics:
            lines.append(f"  Total Detections: {metrics['total_detections']}")
        if "avg_detections_per_image" in metrics:
            lines.append(f"  Avg Detections/Image: {metrics['avg_detections_per_image']:.2f}")
        # 공식 지표
        if "official_oriented_map50_95" in metrics:
            lines.append(f"  [공식 지표] oriented mAP@[0.5:0.95]: {metrics['official_oriented_map50_95']:.4f}")
    
    return "\n".join(lines)


def validate_classify(spec: Dict, log_lines: List[str]) -> Dict[str, Any]:
    """Classification validation: ImageNet1k_100 (ImageNet 1000개 클래스 validation 100개 이미지)"""
    model_path = MODEL_DIR / spec["engine"]
    data_path = DATA_DIR / spec["dataset"]
    
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"[{spec['name']}] Validation 시작")
    log_lines.append(f"  Model: {model_path}")
    log_lines.append(f"  Dataset: {data_path}")
    log_lines.append(f"  Note: ImageNet 1000개 클래스 validation 100개 이미지")
    log_lines.append(f"{'='*80}\n")
    
    # ImageNet1k_100 구조: val/0/image.jpg, val/1/image.jpg, ...
    val_dir = data_path / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    # 이미지 수집 (val 폴더 내 클래스별 폴더에서)
    val_images = sorted([p for p in val_dir.rglob("*.jpg")] + 
                       [p for p in val_dir.rglob("*.png")])
    if not val_images:
        raise RuntimeError(f"No images found in {val_dir}")
    
    log_lines.append(f"검증 이미지: {len(val_images)}개")
    log_lines.append("Note: 정확도 계산 없이 추론 시간만 측정합니다.")
    
    # 모델 로드
    model = YOLO(str(model_path))
    
    # Warm-up
    log_lines.append("Warm-up 실행 중...")
    _ = model.predict(
        source=str(val_images[0]),
        imgsz=spec["imgsz"],
        device=0,
        save=False,
        verbose=False,
    )
    synchronize()
    log_lines.append("Warm-up 완료\n")
    
    # 추론 및 시간 측정 (정확도 계산 없이)
    log_lines.append("추론 시간 측정 중...")
    start_time = time.perf_counter()
    
    for img_path in val_images:
        synchronize()  # 이전 작업 완료 대기
        _ = model.predict(
            source=str(img_path),
            imgsz=spec["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        synchronize()  # CUDA runtime 동기화로 TensorRT 작업 완료까지 대기
    
    elapsed_time = time.perf_counter() - start_time
    num_images = len(val_images)
    avg_time_per_image = elapsed_time / num_images if num_images > 0 else 0
    
    log_lines.append(f"전체 추론 시간: {elapsed_time:.3f}초")
    log_lines.append(f"이미지 수: {num_images}")
    log_lines.append(f"평균 이미지당 시간: {avg_time_per_image*1000:.3f}ms")
    log_lines.append("")
    
    return {
        "model": spec["name"],
        "task": spec["task"],
        "dataset": spec["dataset"],
        "total_time": elapsed_time,
        "num_images": num_images,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "metrics": {},  # 메트릭 없음
    }


def validate_detect_seg_pose(spec: Dict, log_lines: List[str]) -> Dict[str, Any]:
    """Detection/Segmentation/Pose validation: coco2017_val 데이터셋 사용 (COCO JSON 형식)"""
    model_path = MODEL_DIR / spec["engine"]
    data_path = DATA_DIR / spec["dataset"]
    
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"[{spec['name']}] Validation 시작")
    log_lines.append(f"  Model: {model_path}")
    log_lines.append(f"  Dataset: COCO 2017 Val (coco2017_val)")
    log_lines.append(f"  Task: {spec['task']}")
    log_lines.append(f"{'='*80}\n")
    
    # 이미지 디렉토리 확인
    images_dir = data_path / "val2017"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # 이미지 파일 수집 (100개로 제한)
    image_files = sorted(list(images_dir.glob("*.jpg")) + 
                        list(images_dir.glob("*.png")))[:100]
    
    log_lines.append(f"이미지 파일 수집: {len(image_files)}개 (최대 100개로 제한)")
    log_lines.append("Note: 정확도 계산 없이 추론 시간만 측정합니다.")
    
    # 모델 로드
    model = YOLO(str(model_path))
    
    # Warm-up
    log_lines.append("Warm-up 실행 중...")
    if image_files:
        _ = model.predict(
            source=str(image_files[0]),
            imgsz=spec["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        synchronize()
    log_lines.append("Warm-up 완료\n")
    
    # 추론 및 시간 측정 (정확도 계산 없이)
    log_lines.append("추론 시간 측정 중...")
    start_time = time.perf_counter()
    
    # 모든 이미지 처리
    for img_file in image_files:
        synchronize()  # 이전 작업 완료 대기
        _ = model.predict(
            source=str(img_file),
            imgsz=spec["imgsz"],
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
    
    return {
        "model": spec["name"],
        "task": spec["task"],
        "dataset": "coco2017_val",
        "total_time": elapsed_time,
        "num_images": num_images,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "metrics": {},  # 메트릭 없음
    }


def validate_obb(spec: Dict, log_lines: List[str]) -> Dict[str, Any]:
    """OBB validation: DOTAv1.5-1024_val"""
    model_path = MODEL_DIR / spec["engine"]
    data_path = DATA_DIR / spec["dataset"]
    
    val_images_dir = data_path / "images" / "val"
    val_labels_dir = data_path / "labels" / "val"
    
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"[{spec['name']}] Validation 시작")
    log_lines.append(f"  Model: {model_path}")
    log_lines.append(f"  Dataset: DOTA v1.5-1024 Val")
    log_lines.append(f"  Images: {val_images_dir}")
    log_lines.append(f"{'='*80}\n")
    
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {val_images_dir}")
    
    # 이미지 수집 (100개로 제한)
    val_images = sorted([f for f in val_images_dir.iterdir()
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])[:100]

    # train 폴더도 확인 (필요시 사용)
    train_images_dir = data_path / "images" / "train"
    train_images = []
    if train_images_dir.exists():
        train_images = sorted([f for f in train_images_dir.iterdir()
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    # val 이미지를 우선 사용, 부족하면 train에서 보충 (최대 100개)
    combined_images = val_images
    if len(combined_images) < 100 and train_images:
        needed = 100 - len(combined_images)
        combined_images = combined_images + train_images[:needed]
    
    # 최종적으로 100개로 제한
    combined_images = combined_images[:100]

    if not combined_images:
        raise RuntimeError(f"No images found in {val_images_dir}")

    log_lines.append(
        f"검증 이미지: {len(combined_images)}개 (최대 100개로 제한) "
        f"(val: {min(len(val_images), 100)}개, train 사용: {max(0, len(combined_images)-len(val_images))}개)"
    )
    log_lines.append("Note: 정확도 계산 없이 추론 시간만 측정합니다.")
    
    # 모델 로드
    model = YOLO(str(model_path))
    
    # Warm-up
    log_lines.append("Warm-up 실행 중...")
    _ = model.predict(
        source=str(val_images[0]),
        imgsz=spec["imgsz"],
        device=0,
        save=False,
        verbose=False,
    )
    synchronize()
    log_lines.append("Warm-up 완료\n")
    
    # 추론 및 시간 측정 (정확도 계산 없이)
    log_lines.append("추론 시간 측정 중...")
    start_time = time.perf_counter()
    
    for img_path in combined_images:
        synchronize()  # 이전 작업 완료 대기
        _ = model.predict(
            source=str(img_path),
            imgsz=spec["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        synchronize()  # CUDA runtime 동기화로 TensorRT 작업 완료까지 대기
    
    elapsed_time = time.perf_counter() - start_time
    total_processed = len(combined_images)
    avg_time_per_image = elapsed_time / total_processed if total_processed > 0 else 0
    
    log_lines.append(f"전체 추론 시간: {elapsed_time:.3f}초")
    log_lines.append(f"이미지 수: {total_processed}")
    log_lines.append(f"평균 이미지당 시간: {avg_time_per_image*1000:.3f}ms")
    log_lines.append("")
    
    return {
        "model": spec["name"],
        "task": spec["task"],
        "dataset": "DOTAv1.5-1024_val",
        "total_time": elapsed_time,
        "num_images": total_processed,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "metrics": {},  # 메트릭 없음
    }


def validate_model(spec: Dict, log_lines: List[str]) -> Dict[str, Any]:
    """
    모델 validation 수행 및 메트릭 수집
    
    주의: 이 함수는 순차적으로 호출되어야 하며, 병렬 실행되면 안 됩니다.
    각 모델의 validation이 완전히 끝난 후에만 다음 모델로 진행합니다.
    """
    if spec["task"] == "classify":
        return validate_classify(spec, log_lines)
    elif spec["task"] == "obb":
        return validate_obb(spec, log_lines)
    else:
        # detect, pose, segment
        return validate_detect_seg_pose(spec, log_lines)


def main():
    """메인 함수"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULT_DIR / "validation_alone.log"
    txt_path = RESULT_DIR / "validation_alone.txt"
    json_path = RESULT_DIR / "validation_alone.json"
    
    log_lines: List[str] = []
    summary_lines: List[str] = []
    summary_data: List[Dict] = []
    
    log_lines.append("="*80)
    log_lines.append("Validation 시작")
    log_lines.append(f"결과 저장 경로: {RESULT_DIR}")
    log_lines.append("⚠️  중요: 각 engine은 순차적으로 실행됩니다 (병렬 실행 없음)")
    log_lines.append("="*80)
    log_lines.append("")
    
    # 각 engine별로 순차적으로 validation 수행 (병렬 실행 절대 금지)
    for idx, spec in enumerate(MODEL_SPECS, 1):
        try:
            print(f"\n[{idx}/{len(MODEL_SPECS)}] {spec['name']} validation 시작 (순차 실행)")
            summary = validate_model(spec, log_lines)
            summary_data.append(summary)
            print(f"[{idx}/{len(MODEL_SPECS)}] {spec['name']} validation 완료")
            
            # 요약 텍스트 생성
            summary_lines.append(f"\n{'='*80}")
            summary_lines.append(f"[{summary['model']}]")
            summary_lines.append(f"  Dataset: {summary['dataset']}")
            summary_lines.append(f"  Task: {summary['task']}")
            summary_lines.append(f"  전체 시간: {summary['total_time']:.3f}초")
            if summary['num_images'] > 0:
                summary_lines.append(f"  이미지 수: {summary['num_images']}")
                summary_lines.append(f"  평균 시간: {summary['avg_time_per_image_ms']:.3f}ms")
            
            # 메트릭 추가 (없으면 생략)
            if summary.get('metrics'):
                metrics_str = format_metrics(summary['metrics'], summary['task'])
                if metrics_str:
                    summary_lines.append("  메트릭:")
                    for line in metrics_str.split("\n"):
                        if line.strip():
                            summary_lines.append(line)
            
        except Exception as e:
            error_msg = f"[{spec['name']}] 오류 발생: {str(e)}"
            log_lines.append(error_msg)
            log_lines.append("")
            summary_lines.append(error_msg)
            print(f"❌ {error_msg}")
            import traceback
            log_lines.append(traceback.format_exc())
            log_lines.append("")
    
    # 결과 저장
    log_lines.append("\n" + "="*80)
    log_lines.append("Validation 완료")
    log_lines.append("="*80)
    
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    txt_path.write_text("\n".join(summary_lines), encoding="utf-8")
    
    # JSON 형식으로도 저장 (구조화된 데이터)
    json_path.write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"\n✅ Validation 완료!")
    print(f"📄 로그 파일: {log_path}")
    print(f"📄 요약 파일: {txt_path}")
    print(f"📄 JSON 파일: {json_path}")


if __name__ == "__main__":
    main()
