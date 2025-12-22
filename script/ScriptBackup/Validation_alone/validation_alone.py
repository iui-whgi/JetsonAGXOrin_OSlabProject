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


ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "dataset"
RESULT_DIR = ROOT / "result" / "Validation_alone"

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
        "dataset": "coco128",
        "imgsz": 640,
        "task": "detect",
    },
    {
        "name": "yolo11n-pose",
        "engine": "yolo11n-pose.engine",
        "dataset": "coco128",
        "imgsz": 640,
        "task": "pose",
    },
    {
        "name": "yolo11n-seg",
        "engine": "yolo11n-seg.engine",
        "dataset": "coco128",
        "imgsz": 640,
        "task": "segment",
    },
    {
        "name": "yolo11n-obb",
        "engine": "yolo11n-obb.engine",
        "dataset": "DOTA100",
        "imgsz": 640,
        "task": "obb",
    },
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
    else:
        if "precision" in metrics:
            lines.append(f"  Precision: {metrics['precision']:.4f}")
        if "recall" in metrics:
            lines.append(f"  Recall: {metrics['recall']:.4f}")
        if "f1" in metrics:
            lines.append(f"  F1-Score: {metrics['f1']:.4f}")
    
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
    
    # Ground truth: annotation 파일에서 로드 (CSV 또는 JSON)
    # CSV 형식: image_path,class_id,class_name
    csv_file = data_path / "val100.csv"
    gt_dict = {}  # 이미지 경로 -> ImageNet 1000개 클래스 ID
    
    if csv_file.exists():
        import csv
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = row['image_path']  # 예: val/0/013862900598484.jpg
                class_id = int(row['class_id'])  # ImageNet 1000개 클래스 ID (0~999)
                # 전체 경로 생성
                full_path = (data_path / image_path).resolve()
                gt_dict[str(full_path)] = class_id
        log_lines.append(f"Ground truth 로드 완료: {len(gt_dict)}개 (CSV 파일 사용)")
    else:
        # JSON 파일 시도
        json_file = data_path / "annotations_val100.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
            # image_id -> file_name 매핑
            image_id_to_file = {img['id']: img['file_name'] for img in ann_data['images']}
            # annotation에서 image_id -> category_id 매핑
            for ann in ann_data['annotations']:
                image_id = ann['image_id']
                category_id = ann['category_id']  # ImageNet 1000개 클래스 ID
                if image_id in image_id_to_file:
                    file_name = image_id_to_file[image_id]
                    full_path = (data_path / file_name).resolve()
                    gt_dict[str(full_path)] = category_id
            log_lines.append(f"Ground truth 로드 완료: {len(gt_dict)}개 (JSON 파일 사용)")
        else:
            # Annotation 파일이 없으면 폴더명에서 클래스 ID 추출 (fallback)
            log_lines.append("⚠️  Annotation 파일을 찾을 수 없습니다. 폴더명에서 클래스 ID 추출합니다.")
            for img_path in val_images:
                class_folder = img_path.parent.name
                if class_folder.isdigit():
                    gt_dict[str(img_path.resolve())] = int(class_folder)
    
    log_lines.append(f"Ground truth 매핑: {len(gt_dict)}개")
    
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
    
    # 추론 및 시간 측정
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
            imgsz=spec["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        results_list.append(result[0])
        
        # 정확도 계산 (ImageNet 1000개 클래스 ID로 직접 비교)
        if img_str in gt_dict and hasattr(result[0], 'probs') and result[0].probs is not None:
            gt_class_id = gt_dict[img_str]  # ImageNet 1000개 클래스 ID (0~999)
            total_with_gt += 1
            predicted_top1 = int(result[0].probs.top1)
            
            # Top-1 정확도
            if predicted_top1 == gt_class_id:
                correct_top1 += 1
            
            # Top-5 정확도
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
    
    # 메트릭 계산
    top1_accuracy = correct_top1 / total_with_gt if total_with_gt > 0 else 0.0
    top5_accuracy = correct_top5 / total_with_gt if total_with_gt > 0 else 0.0
    
    metrics = {
        "top1": top1_accuracy,
        "top5": top5_accuracy,
    }
    
    log_lines.append(f"전체 추론 시간: {elapsed_time:.3f}초")
    log_lines.append(f"이미지 수: {num_images}")
    log_lines.append(f"평균 이미지당 시간: {avg_time_per_image*1000:.3f}ms")
    if total_with_gt > 0:
        log_lines.append(f"정확도 계산 대상: {total_with_gt}개")
        log_lines.append(f"Top-1 정확도: {correct_top1}/{total_with_gt} = {top1_accuracy:.4f}")
        log_lines.append(f"Top-5 정확도: {correct_top5}/{total_with_gt} = {top5_accuracy:.4f}")
    else:
        log_lines.append("⚠️  정확도 계산을 위한 ground truth를 찾을 수 없습니다.")
    log_lines.append("")
    
    return {
        "model": spec["name"],
        "task": spec["task"],
        "dataset": spec["dataset"],
        "total_time": elapsed_time,
        "num_images": num_images,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "metrics": metrics,
    }


def validate_detect_seg_pose(spec: Dict, log_lines: List[str]) -> Dict[str, Any]:
    """Detection/Segmentation/Pose validation: coco128 데이터셋 사용 (YOLO 형식 labels)"""
    model_path = MODEL_DIR / spec["engine"]
    
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"[{spec['name']}] Validation 시작")
    log_lines.append(f"  Model: {model_path}")
    log_lines.append(f"  Dataset: COCO (coco128)")
    log_lines.append(f"  Task: {spec['task']}")
    log_lines.append(f"{'='*80}\n")
    
    # 이미지 및 labels 디렉토리 확인
    images_dir = DATA_DIR / "coco128" / "images" / "train2017"
    labels_dir = DATA_DIR / "coco128" / "labels" / "train2017"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # 이미지 파일 수집
    image_files = sorted(list(images_dir.glob("*.jpg")) + 
                        list(images_dir.glob("*.png")))
    
    # YOLO 형식 labels에서 ground truth 로드
    # 형식: class_id center_x center_y width height (정규화된 좌표)
    gt_by_image_path = defaultdict(list)
    total_gt_objects = 0
    
    for img_file in image_files:
        label_file = labels_dir / (img_file.stem + ".txt")
        if label_file.exists():
            img_path_str = str(img_file.resolve())
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            center_x = float(parts[1])
                            center_y = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            # YOLO 형식을 COCO bbox 형식으로 변환 (x, y, w, h)
                            # YOLO: center_x, center_y, width, height (정규화)
                            # COCO: x, y, width, height (픽셀 좌표, 정규화 필요)
                            gt_by_image_path[img_path_str].append({
                                "category": class_id,
                                "bbox": [center_x, center_y, width, height],  # 정규화된 좌표
                                "yolo_format": True
                            })
                            total_gt_objects += 1
    
    log_lines.append(f"이미지 파일: {len(image_files)}개")
    log_lines.append(f"Ground-truth 객체: {total_gt_objects}개")
    log_lines.append(f"Ground-truth가 있는 이미지: {len(gt_by_image_path)}개")
    
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
    
    # 추론 및 시간 측정
    log_lines.append("추론 시간 측정 및 정확도 계산 중...")
    start_time = time.perf_counter()
    
    # 배치 크기 제한으로 인해 이미지를 하나씩 처리 (OBB와 동일)
    results_list = []
    total_detected = 0
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    # 모든 이미지 처리
    for img_file in image_files:
        img_path_str = str(img_file.resolve())
        
        # 이미지 하나씩 추론
        result = model.predict(
            source=img_path_str,
            imgsz=spec["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        results_list.append(result[0])
        
        # Ground truth 가져오기
        gt_objects = gt_by_image_path.get(img_path_str, [])
        
        # 결과 분석
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
    
    # 메트릭 계산
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1 = 2 * tp_count / (2 * tp_count + fp_count + fn_count) if (2 * tp_count + fp_count + fn_count) > 0 else 0.0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    num_images = len(results_list)
    avg_time_per_image = elapsed_time / num_images if num_images > 0 else 0
    
    log_lines.append(f"전체 추론 시간: {elapsed_time:.3f}초")
    log_lines.append(f"이미지 수: {num_images}")
    log_lines.append(f"평균 이미지당 시간: {avg_time_per_image*1000:.3f}ms")
    log_lines.append(f"탐지된 객체: {total_detected}개")
    log_lines.append(f"True Positive: {tp_count}개")
    log_lines.append(f"False Positive: {fp_count}개")
    log_lines.append(f"False Negative: {fn_count}개")
    log_lines.append("")
    
    return {
        "model": spec["name"],
        "task": spec["task"],
        "dataset": "coco128",
        "total_time": elapsed_time,
        "num_images": num_images,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "metrics": metrics,
    }


def validate_obb(spec: Dict, log_lines: List[str]) -> Dict[str, Any]:
    """OBB validation: DOTA100"""
    model_path = MODEL_DIR / spec["engine"]
    data_path = DATA_DIR / spec["dataset"]
    
    val_images_dir = data_path / "images" / "val"
    val_labels_dir = data_path / "labels" / "val"
    
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"[{spec['name']}] Validation 시작")
    log_lines.append(f"  Model: {model_path}")
    log_lines.append(f"  Dataset: DOTA100")
    log_lines.append(f"  Images: {val_images_dir}")
    log_lines.append(f"{'='*80}\n")
    
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {val_images_dir}")
    
    # 이미지 수집 (val 폴더 20장 + train 폴더 80장 = 총 100장 사용)
    val_images = sorted([f for f in val_images_dir.iterdir()
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    train_images_dir = data_path / "images" / "train"
    train_images = []
    if train_images_dir.exists():
        train_images = sorted([f for f in train_images_dir.iterdir()
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    combined_images = (val_images + train_images)[:100]

    if not combined_images:
        raise RuntimeError(f"No images found in {val_images_dir}")

    log_lines.append(
        f"검증 이미지: {len(combined_images)}개 "
        f"(val: {len(val_images)}개, train 사용: {max(0, len(combined_images)-len(val_images))}개)"
    )
    
    # Ground-truth 로드
    ground_truth_obbs = []
    if val_labels_dir.exists():
        for label_file in val_labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                ground_truth_obbs.extend(lines)
    
    log_lines.append(f"Ground-truth OBB: {len(ground_truth_obbs)}개")
    
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
    
    # 추론 및 시간 측정
    log_lines.append("추론 시간 측정 중...")
    start_time = time.perf_counter()
    
    # 배치 크기 제한으로 인해 이미지를 하나씩 처리
    results_list = []
    total_detected_obbs = 0
    
    for img_path in combined_images:
        result = model.predict(
            source=str(img_path),
            imgsz=spec["imgsz"],
            device=0,
            save=False,
            verbose=False,
        )
        results_list.append(result[0])
        
        # 결과 분석
        if result[0].obb is not None:
            num_detections = len(result[0].obb)
            total_detected_obbs += num_detections
    
    synchronize()
    elapsed_time = time.perf_counter() - start_time
    
    # 결과 분석
    total_processed = len(results_list)
    
    avg_time_per_image = elapsed_time / total_processed if total_processed > 0 else 0
    
    # 메트릭 (OBB는 정확한 매칭이 복잡하므로 탐지 수만 기록)
    metrics = {
        "total_detections": total_detected_obbs,
        "avg_detections_per_image": total_detected_obbs / total_processed if total_processed > 0 else 0,
    }
    
    log_lines.append(f"전체 추론 시간: {elapsed_time:.3f}초")
    log_lines.append(f"이미지 수: {total_processed}")
    log_lines.append(f"평균 이미지당 시간: {avg_time_per_image*1000:.3f}ms")
    log_lines.append(f"탐지된 OBB: {total_detected_obbs}개")
    log_lines.append(f"이미지당 평균: {metrics['avg_detections_per_image']:.2f}개")
    log_lines.append("")
    
    return {
        "model": spec["name"],
        "task": spec["task"],
        "dataset": "DOTA100",
        "total_time": elapsed_time,
        "num_images": total_processed,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "metrics": metrics,
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
            
            # 메트릭 추가
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
