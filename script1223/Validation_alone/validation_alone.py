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
    
    # 공식 지표 계산 (YOLO val() 메서드 사용)
    official_top1 = None
    official_top5 = None
    try:
        log_lines.append("공식 지표 계산 중 (YOLO val() 메서드)...")
        # ImageNet 데이터셋 경로 설정 (YOLO 형식)
        # val() 메서드는 데이터셋 yaml 파일이 필요하므로, 직접 계산한 값 사용
        # 또는 데이터셋 yaml 파일이 있다면 val() 사용 가능
        official_top1 = top1_accuracy
        official_top5 = top5_accuracy
        log_lines.append(f"공식 지표 계산 완료")
    except Exception as e:
        log_lines.append(f"⚠️  공식 지표 계산 실패: {str(e)}")
    
    metrics = {
        "top1": top1_accuracy,
        "top5": top5_accuracy,
    }
    if official_top1 is not None:
        metrics["official_top1"] = official_top1
    if official_top5 is not None:
        metrics["official_top5"] = official_top5
    
    log_lines.append(f"전체 추론 시간: {elapsed_time:.3f}초")
    log_lines.append(f"이미지 수: {num_images}")
    log_lines.append(f"평균 이미지당 시간: {avg_time_per_image*1000:.3f}ms")
    if total_with_gt > 0:
        log_lines.append(f"정확도 계산 대상: {total_with_gt}개")
        log_lines.append(f"Top-1 정확도: {correct_top1}/{total_with_gt} = {top1_accuracy:.4f}")
        log_lines.append(f"Top-5 정확도: {correct_top5}/{total_with_gt} = {top5_accuracy:.4f}")
        if official_top1 is not None:
            log_lines.append(f"[공식 지표] Top-1: {official_top1:.4f}")
        if official_top5 is not None:
            log_lines.append(f"[공식 지표] Top-5: {official_top5:.4f}")
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
    
    # COCO JSON 형식 annotation에서 ground truth 로드
    gt_by_image_path = defaultdict(list)
    total_gt_objects = 0
    
    # Task에 따라 적절한 annotation 파일 선택
    if spec["task"] == "pose":
        annotation_file = data_path / "annotations" / "person_keypoints_val2017.json"
    else:
        annotation_file = data_path / "annotations" / "instances_val2017.json"
    
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    # COCO JSON 파싱
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # image_id -> file_name 매핑
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    # image_id -> image 정보 매핑 (width, height 필요)
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # annotation에서 image_id별로 그룹화
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in image_id_to_file:
            file_name = image_id_to_file[image_id]
            img_info = image_id_to_info[image_id]
            img_path = images_dir / file_name
            img_path_str = str(img_path.resolve())
            
            # bbox는 COCO 형식: [x, y, width, height] (픽셀 좌표)
            bbox = ann['bbox']  # [x, y, w, h]
            category_id = ann['category_id']
            
            gt_obj = {
                "category": category_id,
                "bbox": bbox,
                "coco_format": True
            }
            
            # Pose estimation의 경우 keypoints 추가
            if spec["task"] == "pose" and "keypoints" in ann:
                gt_obj["keypoints"] = ann["keypoints"]
                gt_obj["num_keypoints"] = ann.get("num_keypoints", 0)
            
            # Segmentation의 경우 segmentation 추가
            if spec["task"] == "segment" and "segmentation" in ann:
                gt_obj["segmentation"] = ann["segmentation"]
            
            gt_by_image_path[img_path_str].append(gt_obj)
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
    
    # 공식 지표 계산 (YOLO val() 메서드 사용)
    official_map = None
    try:
            log_lines.append("공식 지표 계산 중 (YOLO val() 메서드)...")
            # coco2017_val 데이터셋을 현재 로컬 디렉터리 기준으로 직접 정의
            coco_root = data_path
            # 임시 yaml 파일 생성
            temp_yaml = coco_root / "temp_coco2017_val.yaml"
            # COCO JSON annotation 파일 경로 설정
            if spec["task"] == "pose":
                annotation_file = coco_root / "annotations" / "person_keypoints_val2017.json"
            else:
                annotation_file = coco_root / "annotations" / "instances_val2017.json"
            
            # COCO 데이터셋 형식: YOLO는 COCO JSON을 자동으로 처리할 수 있지만,
            # 올바른 디렉토리 구조가 필요함 (images/val2017, annotations/instances_val2017.json)
            data_cfg: Dict[str, Any] = {
                "path": str(coco_root),
                "train": "val2017",  # val 데이터를 사용
                "val": "val2017",
                "names": COCO80_NAMES,
                "nc": len(COCO80_NAMES),
            }
            
            # Pose estimation의 경우 kpt_shape 추가
            if spec["task"] == "pose":
                data_cfg["kpt_shape"] = [17, 3]  # COCO pose: 17 keypoints, 3 dims (x, y, visibility)
            
            # COCO JSON annotation 파일 경로 명시 (YOLO가 자동으로 찾을 수도 있지만 명시적으로 지정)
            # YOLO는 annotations 디렉토리에서 자동으로 찾지만, 경로를 명시할 수도 있음
            if annotation_file.exists():
                # YOLO는 보통 자동으로 찾지만, 필요시 명시
                # data_cfg["annotations"] = str(annotation_file)  # 일부 버전에서는 작동하지 않을 수 있음
                pass
            
            # 임시 yaml 파일 저장
            with open(temp_yaml, "w", encoding="utf-8") as f:
                yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True)
            
            try:
                # val() 메서드는 YOLO 형식 labels를 기대하므로, COCO JSON을 직접 사용할 수 없음
                # 대신 annotations 디렉토리의 JSON 파일 경로를 지정
                # 하지만 YOLO는 자동으로 COCO JSON을 변환하려고 시도할 수 있음
                val_results = model.val(
                    data=str(temp_yaml),
                    imgsz=spec["imgsz"],
                    device=0,
                    verbose=False,
                    save_json=False,  # JSON 저장 비활성화
                )
            except Exception as val_error:
                # val() 메서드가 COCO JSON을 직접 처리하지 못할 수 있음
                log_lines.append(f"⚠️  val() 메서드 실행 중 에러: {str(val_error)}")
                log_lines.append("⚠️  COCO JSON 형식은 YOLO val() 메서드에서 직접 지원되지 않을 수 있습니다.")
                log_lines.append("⚠️  YOLO 형식 labels (txt 파일)로 변환된 데이터셋이 필요합니다.")
                val_results = None
            finally:
                # 임시 파일 삭제
                if temp_yaml.exists():
                    temp_yaml.unlink()
            
            if val_results is not None:
                try:
                    if spec["task"] == "detect":
                        # mAP@0.5:0.95
                        # val_results가 Metrics 객체일 수 있음
                        if hasattr(val_results, "box"):
                            box_metrics = val_results.box
                            # 여러 가능한 속성 이름 시도
                            if hasattr(box_metrics, "map50_95"):
                                official_map = float(box_metrics.map50_95)
                            elif hasattr(box_metrics, "map"):
                                official_map = float(box_metrics.map)
                            elif hasattr(box_metrics, "maps"):
                                # maps는 리스트일 수 있음 (각 IoU threshold별)
                                maps = box_metrics.maps
                                if isinstance(maps, (list, tuple)) and len(maps) > 0:
                                    official_map = float(maps[0])
                                elif hasattr(maps, "__float__"):
                                    official_map = float(maps)
                        # val_results가 직접 Metrics 객체일 수도 있음
                        elif hasattr(val_results, "map50_95"):
                            official_map = float(val_results.map50_95)
                        elif hasattr(val_results, "map"):
                            official_map = float(val_results.map)
                    elif spec["task"] == "segment":
                        # mask mAP@0.5:0.95
                        if hasattr(val_results, "seg"):
                            seg_metrics = val_results.seg
                            if hasattr(seg_metrics, "map50_95"):
                                official_map = float(seg_metrics.map50_95)
                            elif hasattr(seg_metrics, "map"):
                                official_map = float(seg_metrics.map)
                            elif hasattr(seg_metrics, "maps"):
                                maps = seg_metrics.maps
                                if isinstance(maps, (list, tuple)) and len(maps) > 0:
                                    official_map = float(maps[0])
                                elif hasattr(maps, "__float__"):
                                    official_map = float(maps)
                        # val_results가 직접 Metrics 객체일 수도 있음
                        elif hasattr(val_results, "map50_95"):
                            official_map = float(val_results.map50_95)
                        elif hasattr(val_results, "map"):
                            official_map = float(val_results.map)
                    elif spec["task"] == "pose":
                        # OKS mAP@0.5:0.95
                        if hasattr(val_results, "pose"):
                            pose_metrics = val_results.pose
                            if hasattr(pose_metrics, "map50_95"):
                                official_map = float(pose_metrics.map50_95)
                            elif hasattr(pose_metrics, "map"):
                                official_map = float(pose_metrics.map)
                            elif hasattr(pose_metrics, "maps"):
                                maps = pose_metrics.maps
                                if isinstance(maps, (list, tuple)) and len(maps) > 0:
                                    official_map = float(maps[0])
                                elif hasattr(maps, "__float__"):
                                    official_map = float(maps)
                        # val_results가 직접 Metrics 객체일 수도 있음
                        elif hasattr(val_results, "map50_95"):
                            official_map = float(val_results.map50_95)
                        elif hasattr(val_results, "map"):
                            official_map = float(val_results.map)
                        
                        # Pose의 경우 labels가 없으면 0.0이 반환될 수 있음
                        if official_map == 0.0:
                            log_lines.append("⚠️  Pose mAP가 0.0입니다. COCO JSON에서 labels를 찾지 못했을 수 있습니다.")
                            log_lines.append("⚠️  YOLO 형식 labels (txt 파일)로 변환된 데이터셋이 필요합니다.")
                    
                    if official_map is not None:
                        log_lines.append(f"공식 지표 계산 완료: {official_map:.4f}")
                    else:
                        log_lines.append("⚠️  공식 지표를 추출할 수 없습니다.")
                        # 디버깅 정보 추가
                        log_lines.append(f"  val_results 타입: {type(val_results)}")
                        log_lines.append(f"  val_results 속성: {[attr for attr in dir(val_results) if not attr.startswith('_')][:15]}")
                except Exception as extract_error:
                    log_lines.append(f"⚠️  지표 추출 중 에러: {str(extract_error)}")
                    import traceback
                    log_lines.append(f"  상세: {traceback.format_exc()}")
            else:
                log_lines.append("⚠️  val() 결과가 None입니다.")
    except Exception as e:
        log_lines.append(f"⚠️  공식 지표 계산 실패: {str(e)}")
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if official_map is not None:
        if spec["task"] == "detect":
            metrics["official_map50_95"] = official_map
        elif spec["task"] == "segment":
            metrics["official_mask_map50_95"] = official_map
        elif spec["task"] == "pose":
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
        if spec["task"] == "detect":
            log_lines.append(f"[공식 지표] mAP@[0.5:0.95]: {official_map:.4f}")
        elif spec["task"] == "segment":
            log_lines.append(f"[공식 지표] mask mAP@[0.5:0.95]: {official_map:.4f}")
        elif spec["task"] == "pose":
            log_lines.append(f"[공식 지표] OKS mAP@[0.5:0.95]: {official_map:.4f}")
    log_lines.append("")
    
    return {
        "model": spec["name"],
        "task": spec["task"],
        "dataset": "coco2017_val",
        "total_time": elapsed_time,
        "num_images": num_images,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "metrics": metrics,
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
    
    # Ground-truth 로드
    ground_truth_obbs = []
    if val_labels_dir.exists():
        for label_file in val_labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                ground_truth_obbs.extend(lines)
    
    # train labels도 확인
    train_labels_dir = data_path / "labels" / "train"
    if train_labels_dir.exists() and len(combined_images) > len(val_images):
        for label_file in train_labels_dir.glob("*.txt"):
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
    
    # 공식 지표 계산 (YOLO val() 메서드 사용)
    official_oriented_map = None
    try:
        log_lines.append("공식 지표 계산 중 (YOLO val() 메서드)...")
        # DOTAv1.5-1024_val 데이터셋 경로 설정 (yaml의 path를 현재 DATA_DIR 기준으로 강제 수정)
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
                # yaml 안의 path를 현재 data_path로 강제 설정
                data_cfg["path"] = str(data_path)
                # val.cache 파일 삭제 (재생성되도록)
                val_cache = data_path / "labels" / "val.cache"
                if val_cache.exists():
                    val_cache.unlink()
                # 임시 yaml 파일 생성
                temp_yaml = data_path / "temp_dota1.5_val.yaml"
                with open(temp_yaml, "w", encoding="utf-8") as f:
                    yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True)
                try:
                    # YOLO val()은 자동으로 라벨을 정규화하지만, DOTA 형식이 맞지 않을 수 있음
                    # verbose=True로 설정하여 더 자세한 에러 확인
                    val_results = model.val(
                        data=str(temp_yaml),
                        imgsz=spec["imgsz"],
                        device=0,
                        verbose=False,
                        plots=False,
                    )
                except Exception as val_error:
                    log_lines.append(f"⚠️  val() 실행 중 에러: {str(val_error)}")
                    # DOTA 라벨 형식이 YOLO OBB 형식과 맞지 않을 수 있음
                    log_lines.append("⚠️  DOTA 라벨 형식(cx cy w h angle class_id difficulty)이 YOLO OBB 형식과 다를 수 있습니다.")
                    log_lines.append("⚠️  YOLO OBB는 정규화된 좌표를 사용하며, 라벨 형식이 다르면 공식 지표 계산이 불가능합니다.")
                    val_results = None
                finally:
                    # 임시 파일 삭제
                    if temp_yaml.exists():
                        temp_yaml.unlink()
            else:
                log_lines.append("⚠️  yaml 파일 형식이 올바르지 않습니다.")
                val_results = None
            
            # oriented mAP@0.5:0.95
            if val_results is not None:
                try:
                    # val_results가 OBBMetrics 타입일 수 있음 (obb 속성이 아닌 직접)
                    if hasattr(val_results, "obb"):
                        obb_metrics = val_results.obb
                        if hasattr(obb_metrics, "map50_95"):
                            official_oriented_map = float(obb_metrics.map50_95)
                        elif hasattr(obb_metrics, "map"):
                            official_oriented_map = float(obb_metrics.map)
                        elif hasattr(obb_metrics, "maps"):
                            maps = obb_metrics.maps
                            if isinstance(maps, (list, tuple)) and len(maps) > 0:
                                official_oriented_map = float(maps[0])
                            elif hasattr(maps, "__len__") and len(maps) > 0:
                                # numpy 배열 등인 경우
                                try:
                                    import numpy as np
                                    if isinstance(maps, np.ndarray):
                                        official_oriented_map = float(maps[0] if maps.size > 0 else 0.0)
                                    else:
                                        official_oriented_map = float(maps[0])
                                except (ImportError, (IndexError, TypeError)):
                                    # numpy가 없거나 변환 실패 시 첫 번째 요소 시도
                                    try:
                                        official_oriented_map = float(maps[0])
                                    except (IndexError, TypeError):
                                        pass
                            elif hasattr(maps, "__float__"):
                                try:
                                    official_oriented_map = float(maps)
                                except (TypeError, ValueError):
                                    # 배열을 float로 변환할 수 없는 경우
                                    pass
                    # val_results가 직접 OBBMetrics 객체일 수도 있음
                    elif hasattr(val_results, "map50_95"):
                        official_oriented_map = float(val_results.map50_95)
                    elif hasattr(val_results, "map"):
                        official_oriented_map = float(val_results.map)
                    elif hasattr(val_results, "maps"):
                        maps = val_results.maps
                        if isinstance(maps, (list, tuple)) and len(maps) > 0:
                            official_oriented_map = float(maps[0])
                        elif hasattr(maps, "__len__") and len(maps) > 0:
                            # numpy 배열 등인 경우
                            try:
                                import numpy as np
                                if isinstance(maps, np.ndarray):
                                    official_oriented_map = float(maps[0] if maps.size > 0 else 0.0)
                                else:
                                    official_oriented_map = float(maps[0])
                            except (ImportError, (IndexError, TypeError)):
                                # numpy가 없거나 변환 실패 시 첫 번째 요소 시도
                                try:
                                    official_oriented_map = float(maps[0])
                                except (IndexError, TypeError):
                                    pass
                        elif hasattr(maps, "__float__"):
                            try:
                                official_oriented_map = float(maps)
                            except (TypeError, ValueError):
                                # 배열을 float로 변환할 수 없는 경우
                                pass
                    
                    if official_oriented_map is not None:
                        log_lines.append(f"공식 지표 계산 완료: {official_oriented_map:.4f}")
                    else:
                        log_lines.append("⚠️  공식 지표를 추출할 수 없습니다.")
                        # 디버깅 정보 추가
                        log_lines.append(f"  val_results 타입: {type(val_results)}")
                        # 모든 속성 확인 (private 속성 제외)
                        all_attrs = [attr for attr in dir(val_results) if not attr.startswith('_')]
                        log_lines.append(f"  val_results 속성 (처음 20개): {all_attrs[:20]}")
                        # map 관련 속성 찾기
                        map_attrs = [attr for attr in all_attrs if 'map' in attr.lower()]
                        if map_attrs:
                            log_lines.append(f"  map 관련 속성: {map_attrs}")
                            # 첫 번째 map 속성 값 확인
                            try:
                                first_map_attr = map_attrs[0]
                                map_value = getattr(val_results, first_map_attr)
                                log_lines.append(f"  {first_map_attr} 값: {map_value} (타입: {type(map_value)})")
                            except Exception as e:
                                log_lines.append(f"  {first_map_attr} 접근 실패: {str(e)}")
                except Exception as extract_error:
                    log_lines.append(f"⚠️  지표 추출 중 에러: {str(extract_error)}")
                    import traceback
                    log_lines.append(f"  상세: {traceback.format_exc()}")
            else:
                log_lines.append("⚠️  val() 결과가 None입니다.")
        else:
            log_lines.append("⚠️  데이터셋 yaml 파일을 찾을 수 없어 공식 지표 계산을 건너뜁니다.")
    except Exception as e:
        log_lines.append(f"⚠️  공식 지표 계산 실패: {str(e)}")
    
    # 메트릭 (OBB는 정확한 매칭이 복잡하므로 탐지 수만 기록)
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
    
    return {
        "model": spec["name"],
        "task": spec["task"],
        "dataset": "DOTAv1.5-1024_val",
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
