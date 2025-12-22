#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/home/gpu-agx/zoo/script/Validation_objectdetect/validate_pose.py
coco2017_val 데이터셋을 사용하여 yolo11n-pose.engine 모델의 validation 평가를 수행합니다.
annotations2의 pose labels를 사용합니다.
"""
import time
from pathlib import Path
from typing import List, Dict, Any
import json
from collections import defaultdict
import logging
from datetime import datetime
import shutil
import os

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO
import yaml

# Docker 환경을 고려하여 경로 설정
ROOT = Path("/ultralytics/zoo")
SCRIPT_DIR = ROOT / "script" / "Validation_objectdetect"
MODEL_PATH = ROOT / "model" / "yolo11n-pose.engine"
DATASET_PATH = ROOT / "dataset" / "coco2017_val"
IMGSZ = 640
TASK = "pose"

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
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()

def setup_logging(log_dir):
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
    """Pose estimation validation: annotations2의 pose labels 사용"""
    
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
    
    images_dir = DATASET_PATH / "val2017"
    if not images_dir.exists():
        error_msg = f"Images directory not found: {images_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # annotations2의 pose labels 디렉토리
    labels_pose_dir = DATASET_PATH / "labels" / "person_keypoints_val2017"
    if not labels_pose_dir.exists():
        error_msg = f"Pose labels directory not found: {labels_pose_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # labels_pose 디렉토리의 모든 label 파일 로드
    pose_label_files = {f.stem for f in labels_pose_dir.glob("*.txt")}
    logger.info(f"Pose 형식 labels 사용: {labels_pose_dir}")
    log_lines.append(f"Pose 형식 labels 사용: {labels_pose_dir}")
    log_lines.append(f"  총 {len(pose_label_files)}개 파일")
    
    # 이미지 파일 수집 및 필터링 (labels_pose에 해당하는 이미지만)
    all_image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    image_files = [
        img_file for img_file in all_image_files
        if img_file.stem in pose_label_files
    ]
    
    logger.info(f"이미지 파일 수집: {len(image_files)}개 (Pose labels와 매칭되는 파일만)")
    log_lines.append(f"이미지 파일 수집: {len(image_files)}개 (Pose labels와 매칭되는 파일만)")
    
    if not image_files:
        error_msg = "매칭되는 이미지 파일이 없습니다. Pose validation을 수행할 수 없습니다."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("모델 로드 중...")
    model = YOLO(str(MODEL_PATH))
    logger.info("모델 로드 완료")
    
    logger.info("Warm-up 실행 중...")
    log_lines.append("Warm-up 실행 중...")
    if image_files:
        _ = model.predict(source=str(image_files[0]), imgsz=IMGSZ, device=0, save=False, verbose=False)
        synchronize()
    logger.info("Warm-up 완료")
    log_lines.append("Warm-up 완료\n")
    
    # 공식 지표 계산 (YOLO val() 메서드 사용)
    official_map = None
    try:
        logger.info("공식 지표 계산 중 (YOLO val() 메서드)...")
        log_lines.append("공식 지표 계산 중 (YOLO val() 메서드)...")
        
        coco_root = DATASET_PATH
        temp_yaml = coco_root / "temp_coco2017_val_pose.yaml"
        
        # YOLO 형식 데이터셋 설정
        data_cfg = {
            "path": str(coco_root),
            "train": "val2017",
            "val": "val2017",
            "names": COCO80_NAMES,
            "nc": len(COCO80_NAMES),
            "kpt_shape": [17, 3],  # COCO pose: 17 keypoints, 3 dims (x, y, visibility)
        }
        
        # labels 디렉토리 설정 (labels/person_keypoints_val2017를 labels/val2017로 심볼릭 링크)
        target_labels_dir = coco_root / "labels" / "val2017"
        target_labels_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # 기존 labels가 일반 디렉토리면 삭제 후 심볼릭 링크 생성
        if target_labels_dir.exists() and not target_labels_dir.is_symlink():
            try:
                shutil.rmtree(str(target_labels_dir))
                logger.info(f"기존 labels 디렉토리 삭제: {target_labels_dir}")
            except Exception as e:
                logger.warning(f"기존 labels 삭제 실패: {e}")
        
        # 기존 심볼릭 링크가 있다면 삭제
        if target_labels_dir.exists() and target_labels_dir.is_symlink():
            target_labels_dir.unlink()
            logger.info(f"기존 심볼릭 링크 삭제: {target_labels_dir}")
        
        # 새로운 심볼릭 링크 생성
        if not target_labels_dir.exists():
            os.symlink(str(labels_pose_dir), str(target_labels_dir))
            logger.info(f"심볼릭 링크 생성: {target_labels_dir} -> {labels_pose_dir}")
            log_lines.append(f"심볼릭 링크 생성: {target_labels_dir} -> {labels_pose_dir}")
        
        # 필터링된 이미지만 포함하는 임시 디렉토리 생성
        import tempfile
        temp_images_dir = None
        temp_labels_dir = None
        try:
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
                label_file = labels_pose_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    dest_label = temp_labels_dir / f"{img_file.stem}.txt"
                    shutil.copy2(str(label_file), str(dest_label))
                    label_count += 1
            
            logger.info(f"필터링된 이미지 {len(image_files)}개, labels {label_count}개를 임시 디렉토리로 복사 완료")
            
            # yaml 파일의 이미지 경로를 임시 디렉토리로 수정
            data_cfg["path"] = str(temp_base)
            data_cfg["val"] = "val2017"
            
            # 캐시 파일 삭제
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
            
            # yaml 파일 저장
            with open(temp_yaml, "w", encoding="utf-8") as f:
                yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True)
            
            val_results = model.val(
                data=str(temp_yaml),
                imgsz=IMGSZ,
                device=0,
                verbose=False,
                save_json=False,
                plots=False,
            )
            
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
        finally:
            # 임시 디렉토리 정리
            if temp_images_dir and temp_images_dir.parent.exists():
                shutil.rmtree(str(temp_images_dir.parent))
                logger.info(f"임시 디렉토리 삭제: {temp_images_dir.parent}")
            
            # 임시 yaml 파일 삭제
            if temp_yaml.exists():
                temp_yaml.unlink()
    except Exception as e:
        logger.error(f"공식 지표 계산 실패: {str(e)}")
        log_lines.append(f"⚠️  공식 지표 계산 실패: {str(e)}")
        import traceback
        log_lines.append(traceback.format_exc())
    
    metrics = {}
    if official_map is not None:
        metrics["official_oks_map50_95"] = official_map
    
    num_images = len(image_files)
    
    logger.info(f"이미지 수: {num_images}")
    logger.info(f"Label 파일 수: {len(pose_label_files)}")
    if official_map is not None:
        logger.info(f"[공식 지표] OKS mAP@[0.5:0.95]: {official_map:.4f}")
    
    log_lines.append(f"이미지 수: {num_images}")
    log_lines.append(f"Label 파일 수: {len(pose_label_files)}")
    if official_map is not None:
        log_lines.append(f"[공식 지표] OKS mAP@[0.5:0.95]: {official_map:.4f}")
    log_lines.append("")
    
    return {"model": "yolo11n-pose", "task": TASK, "dataset": "coco2017_val", "num_images": num_images, "num_label_files": len(pose_label_files), "metrics": metrics}

def main():
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
        summary_lines.append(f"  Label 파일 수: {summary['num_label_files']}")
        
        metrics = summary['metrics']
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
    
    # 파일 저장 시도
    try:
        txt_path = SCRIPT_DIR / "validate_pose_summary.txt"
        full_log_path = SCRIPT_DIR / "validate_pose_full.log"
        
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
        logger.error(f"결과 파일 저장 중 오류 발생: {str(e)}")
        print(f"❌ 결과 파일 저장 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
