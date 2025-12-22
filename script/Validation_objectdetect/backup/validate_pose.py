#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
/home/gpu-agx/zoo/script/Validation_objectdetect/validate_pose.py
coco2017_val 데이터셋을 사용하여 yolo11n-pose.engine 모델의 validation 평가를 수행합니다.
기존 YOLO pose 형식 labels (labels_pose/val2017)를 사용하여 공식 성능 지표를 계산합니다.
로그는 같은 디렉토리에 저장됩니다.
'''
import time
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO
import yaml

# 경로 설정 (Docker 컨테이너 내부 경로도 지원)
if Path("/ultralytics/zoo").exists():
    ROOT = Path("/ultralytics/zoo")
else:
    ROOT = Path("/home/gpu-agx/zoo")

SCRIPT_DIR = ROOT / "script" / "Validation_objectdetect"
MODEL_PATH = ROOT / "model" / "yolo11n-pose.engine"
DATASET_PATH = ROOT / "dataset" / "coco2017_val"
IMGSZ = 640
TASK = "pose"

# COCO 80 classes
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

def synchronize():
    """GPU 동기화"""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()

def setup_logging(log_dir: Path):
    """로깅 설정"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"validate_pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), log_file

def validate_pose(logger: logging.Logger, log_lines: List[str]) -> Dict[str, Any]:
    """Pose estimation validation: coco2017_val 데이터셋 사용 (기존 YOLO pose 형식 labels 활용)"""
    
    logger.info("="*80)
    logger.info("YOLO11n-Pose Validation 시작")
    logger.info(f"  Model: {MODEL_PATH}")
    logger.info(f"  Dataset: COCO 2017 Val (coco2017_val)")
    logger.info(f"  Task: {TASK}")
    logger.info("="*80)
    
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"YOLO11n-Pose Validation 시작")
    log_lines.append(f"  Model: {MODEL_PATH}")
    log_lines.append(f"  Dataset: COCO 2017 Val (coco2017_val)")
    log_lines.append(f"  Task: {TASK}")
    log_lines.append(f"{'='*80}\n")
    
    # 이미지 디렉토리 확인
    images_dir = DATASET_PATH / "val2017"
    if not images_dir.exists():
        error_msg = f"Images directory not found: {images_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Pose 형식 labels 디렉토리 확인 (먼저 정의)
    labels_pose_dir = DATASET_PATH / "labels_pose" / "val2017"
    labels_dir = DATASET_PATH / "labels" / "val2017"
    
    # labels_pose 우선 사용, 없으면 일반 labels 사용
    if labels_pose_dir.exists() and labels_pose_dir.is_dir():
        labels_dir = labels_pose_dir
        logger.info(f"Pose 형식 labels 사용: {labels_dir}")
        log_lines.append(f"Pose 형식 labels 사용: {labels_dir}")
    else:
        logger.info(f"일반 labels 사용: {labels_dir}")
        log_lines.append(f"일반 labels 사용: {labels_dir}")
    
    # 이미지 파일 수집
    all_image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    logger.info(f"전체 이미지 파일: {len(all_image_files)}개")
    log_lines.append(f"전체 이미지 파일: {len(all_image_files)}개")
    
    # Label이 있는 이미지만 필터링 (pose validation을 위해 필수)
    label_stems = {f.stem for f in labels_dir.glob("*.txt")} if labels_dir.exists() else set()
    image_files = [img for img in all_image_files if img.stem in label_stems]
    logger.info(f"Label이 있는 이미지: {len(image_files)}개 (전체 {len(all_image_files)}개 중)")
    log_lines.append(f"Label이 있는 이미지: {len(image_files)}개 (전체 {len(all_image_files)}개 중)")
    
    if len(image_files) == 0:
        error_msg = f"Label이 있는 이미지가 없습니다. labels 디렉토리를 확인하세요: {labels_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # labels_pose 우선 사용, 없으면 일반 labels 사용
    if labels_pose_dir.exists() and labels_pose_dir.is_dir():
        labels_dir = labels_pose_dir
        logger.info(f"Pose 형식 labels 사용: {labels_dir}")
        log_lines.append(f"Pose 형식 labels 사용: {labels_dir}")
    else:
        logger.info(f"일반 labels 사용: {labels_dir}")
        log_lines.append(f"일반 labels 사용: {labels_dir}")
    
    # Label 파일 확인
    label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    logger.info(f"YOLO label 파일: {len(label_files)}개")
    log_lines.append(f"YOLO label 파일: {len(label_files)}개")
    
    # 모델 로드
    logger.info("모델 로드 중...")
    model = YOLO(str(MODEL_PATH))
    logger.info("모델 로드 완료")
    log_lines.append("모델 로드 완료")
    
    # Warm-up
    logger.info("Warm-up 실행 중...")
    log_lines.append("Warm-up 실행 중...")
    if image_files:
        _ = model.predict(source=str(image_files[0]), imgsz=IMGSZ, device=0, save=False, verbose=False)
        synchronize()
    logger.info("Warm-up 완료")
    log_lines.append("Warm-up 완료\n")
    
    # 공식 지표 계산 (YOLO val() 메서드 사용)
    official_map = None
    logger.info("공식 지표 계산 중 (YOLO val() 메서드)...")
    log_lines.append("공식 지표 계산 중 (YOLO val() 메서드)...")
    
    # 변수 초기화
    coco_root = DATASET_PATH
    temp_yaml = coco_root / "temp_coco2017_val_pose.yaml"
    annotations_backup = None
    
    try:
        
        # labels_pose를 labels로 심볼릭 링크 (YOLO가 인식하도록)
        import os
        target_labels = coco_root / "labels" / "val2017"
        
        if labels_dir != target_labels and labels_dir.exists():
            # 기존 labels 백업
            if target_labels.exists() and not target_labels.is_symlink():
                backup_labels = coco_root / "labels_backup" / "val2017"
                backup_labels.parent.mkdir(parents=True, exist_ok=True)
                if not backup_labels.exists():
                    import shutil
                    shutil.move(str(target_labels), str(backup_labels))
                    logger.info(f"기존 labels 백업: {backup_labels}")
            
            # 심볼릭 링크 생성 (부모 디렉토리 먼저 생성)
            target_labels.parent.mkdir(parents=True, exist_ok=True)
            if target_labels.exists() and target_labels.is_symlink():
                target_labels.unlink()
            if not target_labels.exists():
                os.symlink(str(labels_dir), str(target_labels))
                logger.info(f"심볼릭 링크 생성: {target_labels} -> {labels_dir}")
                log_lines.append(f"심볼릭 링크 생성: {target_labels} -> {labels_dir}")
        
        # YOLO 형식 데이터셋 설정
        # labels_pose를 직접 사용하도록 경로 지정
        # YOLO는 path/labels/val2017 경로를 찾으므로, labels_pose를 labels로 사용
        data_cfg = {
            "path": str(coco_root),
            "train": "val2017",
            "val": "val2017",
            "names": COCO80_NAMES,
            "nc": len(COCO80_NAMES),
            "kpt_shape": [17, 3],  # COCO pose: 17 keypoints, 3 dims (x, y, visibility)
        }
        
        # labels_pose를 labels로 심볼릭 링크 (반드시 실행)
        if labels_dir != target_labels and labels_dir.exists():
            # 기존 labels가 일반 디렉토리면 삭제 후 심볼릭 링크 생성
            if target_labels.exists() and not target_labels.is_symlink():
                # 기존 디렉토리는 백업되어 있으므로 삭제
                try:
                    import shutil
                    shutil.rmtree(str(target_labels))
                    logger.info(f"기존 labels 디렉토리 삭제: {target_labels}")
                except Exception as e:
                    logger.warning(f"기존 labels 삭제 실패: {e}")
            
            # 심볼릭 링크 생성 (부모 디렉토리 먼저 생성)
            target_labels.parent.mkdir(parents=True, exist_ok=True)
            if not target_labels.exists():
                os.symlink(str(labels_dir), str(target_labels))
                logger.info(f"심볼릭 링크 생성: {target_labels} -> {labels_dir}")
                log_lines.append(f"심볼릭 링크 생성: {target_labels} -> {labels_dir}")
            elif target_labels.is_symlink():
                # 이미 심볼릭 링크가 있으면 확인
                if target_labels.readlink() != labels_dir:
                    target_labels.unlink()
                    os.symlink(str(labels_dir), str(target_labels))
                    logger.info(f"심볼릭 링크 업데이트: {target_labels} -> {labels_dir}")
                    log_lines.append(f"심볼릭 링크 업데이트: {target_labels} -> {labels_dir}")
                else:
                    logger.info(f"심볼릭 링크 이미 올바름: {target_labels} -> {labels_dir}")
                    log_lines.append(f"심볼릭 링크 이미 올바름: {target_labels} -> {labels_dir}")
        
        # annotations 디렉토리 임시 백업 (YOLO가 labels만 사용하도록)
        annotations_backup = None
        annotations_dir = coco_root / "annotations"
        if annotations_dir.exists():
            annotations_backup = coco_root / "annotations_backup"
            if not annotations_backup.exists():
                annotations_dir.rename(annotations_backup)
                logger.info(f"annotations 디렉토리 백업: {annotations_backup}")
        
        # 캐시 파일 삭제
        cache_file = coco_root / "val2017.cache"
        if cache_file.exists():
            cache_file.unlink()
        
        # 임시 yaml 파일 저장
        with open(temp_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True)
        
        # YOLO val() 메서드 실행
        # 필터링된 이미지만 포함하는 임시 디렉토리 생성
        import tempfile
        import shutil
        temp_images_dir = None
        temp_labels_dir = None
        try:
            # 임시 디렉토리 생성 (YOLO 형식: path/val2017 구조)
            temp_base = Path(tempfile.mkdtemp(prefix="pose_val_"))
            temp_images_dir = temp_base / "val2017"
            temp_images_dir.mkdir(parents=True, exist_ok=True)
            temp_labels_dir = temp_base / "labels" / "val2017"
            temp_labels_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"임시 디렉토리 생성: {temp_base}")
            
            # 필터링된 이미지만 임시 디렉토리로 복사
            for img_file in image_files:
                dest_file = temp_images_dir / img_file.name
                shutil.copy2(str(img_file), str(dest_file))
            
            # 필터링된 이미지에 해당하는 labels만 복사
            label_count = 0
            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    dest_label = temp_labels_dir / f"{img_file.stem}.txt"
                    shutil.copy2(str(label_file), str(dest_label))
                    label_count += 1
            
            logger.info(f"필터링된 이미지 {len(image_files)}개, labels {label_count}개를 임시 디렉토리로 복사 완료")
            
            # yaml 파일의 이미지 경로를 임시 디렉토리로 수정
            data_cfg["path"] = str(temp_base)  # 임시 디렉토리
            data_cfg["val"] = "val2017"  # val2017 서브디렉토리
            
            # yaml 파일 다시 저장
            with open(temp_yaml, "w", encoding="utf-8") as f:
                yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True)
            
            # 캐시 파일 삭제 (labels 인식 문제 해결)
            cache_file = temp_base / "val2017.cache"
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"캐시 파일 삭제: {cache_file}")
            
            # labels 디렉토리 존재 확인
            if not temp_labels_dir.exists() or len(list(temp_labels_dir.glob("*.txt"))) == 0:
                error_msg = f"Labels 디렉토리가 비어있거나 존재하지 않습니다: {temp_labels_dir}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.info(f"Labels 파일 수: {len(list(temp_labels_dir.glob('*.txt')))}개")
            
            val_results = model.val(
                data=str(temp_yaml),
                imgsz=IMGSZ,
                device=0,
                verbose=False,
                save_json=False,
                plots=False,
            )
        finally:
            # 임시 디렉토리 정리
            if temp_images_dir and temp_images_dir.parent.exists():
                shutil.rmtree(str(temp_images_dir.parent))
                logger.info(f"임시 디렉토리 삭제: {temp_images_dir.parent}")
        
        # 결과 추출
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
                    logger.info(f"공식 지표 계산 완료: OKS mAP@[0.5:0.95] = {official_map:.4f}")
                    log_lines.append(f"공식 지표 계산 완료: OKS mAP@[0.5:0.95] = {official_map:.4f}")
                else:
                    logger.warning("공식 지표를 추출할 수 없습니다.")
                    log_lines.append("⚠️  공식 지표를 추출할 수 없습니다.")
            except Exception as extract_error:
                logger.warning(f"지표 추출 중 에러: {str(extract_error)}")
                log_lines.append(f"⚠️  지표 추출 중 에러: {str(extract_error)}")
        else:
            logger.warning("val() 결과가 None입니다.")
            log_lines.append("⚠️  val() 결과가 None입니다.")
    
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"공식 지표 계산 실패: {error_msg}")
        log_lines.append(f"⚠️  공식 지표 계산 실패: {error_msg}")
        official_map = None
    
    finally:
        # annotations 디렉토리 복원
        if annotations_backup and annotations_backup.exists():
            annotations_restored = coco_root / "annotations"
            if not annotations_restored.exists():
                annotations_backup.rename(annotations_restored)
                logger.info(f"annotations 디렉토리 복원: {annotations_restored}")
        
        # 임시 파일 삭제
        if temp_yaml.exists():
            temp_yaml.unlink()
    
    # 결과 정리
    metrics = {}
    if official_map is not None:
        metrics["official_oks_map50_95"] = official_map
    
    return {
        "model": "yolo11n-pose",
        "task": TASK,
        "dataset": "coco2017_val",
        "num_images": len(image_files),
        "num_labels": len(label_files),
        "metrics": metrics,
    }

def main():
    """메인 함수"""
    logger, log_file = setup_logging(SCRIPT_DIR)
    logger.info("="*80)
    logger.info("YOLO11n-Pose Validation 시작")
    logger.info(f"결과 저장 경로: {SCRIPT_DIR}")
    logger.info("="*80)
    
    log_lines: List[str] = []
    summary_lines: List[str] = []
    
    log_lines.append("="*80)
    log_lines.append("YOLO11n-Pose Validation 시작")
    log_lines.append(f"결과 저장 경로: {SCRIPT_DIR}")
    log_lines.append("="*80)
    log_lines.append("")
    
    try:
        summary = validate_pose(logger, log_lines)
        
        summary_lines.append(f"\n{'='*80}")
        summary_lines.append(f"[{summary['model']}]")
        summary_lines.append(f"  Dataset: {summary['dataset']}")
        summary_lines.append(f"  Task: {summary['task']}")
        summary_lines.append(f"  이미지 수: {summary['num_images']}")
        summary_lines.append(f"  Label 파일 수: {summary['num_labels']}")
        
        metrics = summary['metrics']
        if metrics:
            summary_lines.append("  메트릭:")
            if "official_oks_map50_95" in metrics:
                summary_lines.append(f"    [공식 지표] OKS mAP@[0.5:0.95]: {metrics['official_oks_map50_95']:.4f}")
        
        logger.info("Validation 완료")
        log_lines.append("\n" + "="*80)
        log_lines.append("Validation 완료")
        log_lines.append("="*80)
    
    except Exception as e:
        error_msg = f"오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True)
        log_lines.append(error_msg)
        log_lines.append("")
        import traceback
        log_lines.append(traceback.format_exc())
        log_lines.append("")
        summary_lines.append(error_msg)
    
    # 결과 파일 저장
    txt_path = SCRIPT_DIR / "validate_pose_summary.txt"
    full_log_path = SCRIPT_DIR / "validate_pose_full.log"
    
    try:
        txt_path.write_text("\n".join(summary_lines), encoding="utf-8")
        full_log_path.write_text("\n".join(log_lines), encoding="utf-8")
        logger.info(f"✅ Validation 완료!")
        logger.info(f"📄 로그 파일: {log_file}")
        logger.info(f"📄 전체 로그 파일: {full_log_path}")
        logger.info(f"📄 요약 파일: {txt_path}")
        print(f"\n✅ Validation 완료!")
        print(f"📄 로그 파일: {log_file}")
        print(f"📄 전체 로그 파일: {full_log_path}")
        print(f"📄 요약 파일: {txt_path}")
    except Exception as e:
        logger.error(f"파일 저장 실패: {str(e)}")
        print(f"파일 저장 실패: {str(e)}")

if __name__ == "__main__":
    main()
