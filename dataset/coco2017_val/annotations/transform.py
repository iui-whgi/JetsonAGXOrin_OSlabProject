import tempfile
import shutil
from pathlib import Path
from ultralytics.data.converter import convert_coco

# person_keypoints_val2017.json 변환 (pose용)
print("=" * 80)
print("1. person_keypoints_val2017.json 변환 시작 (pose용)")
print("=" * 80)

temp_dir_pose = Path(tempfile.mkdtemp(prefix="coco_pose_"))
source_json_pose = Path("/home/gpu-agx/zoo/dataset/coco2017_val/annotations/person_keypoints_val2017.json")
dest_json_pose = temp_dir_pose / "person_keypoints_val2017.json"

shutil.copy2(str(source_json_pose), str(dest_json_pose))
print(f"임시 디렉토리 생성: {temp_dir_pose}")

# convert_coco 실행 (pose용)
result_pose = convert_coco(
    labels_dir=str(temp_dir_pose),
    save_dir="/home/gpu-agx/zoo/dataset/coco2017_val/annotations2",
    use_keypoints=True,
)

# 결과 디렉토리 찾기
result_dirs = sorted(Path("/home/gpu-agx/zoo/dataset/coco2017_val").glob("annotations*"), 
                     key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
if result_dirs:
    latest_dir_pose = result_dirs[0]
    if latest_dir_pose.name != "annotations2" and latest_dir_pose.exists():
        # annotations2로 복사
        dest = Path("/home/gpu-agx/zoo/dataset/coco2017_val/annotations2")
        if dest.exists():
            shutil.rmtree(str(dest))
        shutil.copytree(str(latest_dir_pose), str(dest))
        print(f"{latest_dir_pose.name}의 내용을 annotations2로 복사 완료 (pose)")

# 임시 디렉토리 삭제
shutil.rmtree(str(temp_dir_pose))
print("person_keypoints_val2017.json 변환 완료\n")

# instances_val2017.json 변환 (detection용)
print("=" * 80)
print("2. instances_val2017.json 변환 시작 (detection용)")
print("=" * 80)

temp_dir_det = Path(tempfile.mkdtemp(prefix="coco_detection_"))
source_json_det = Path("/home/gpu-agx/zoo/dataset/coco2017_val/annotations/instances_val2017.json")
dest_json_det = temp_dir_det / "instances_val2017.json"

shutil.copy2(str(source_json_det), str(dest_json_det))
print(f"임시 디렉토리 생성: {temp_dir_det}")

# convert_coco 실행 (detection용)
result_det = convert_coco(
    labels_dir=str(temp_dir_det),
    save_dir="/home/gpu-agx/zoo/dataset/coco2017_val/annotations2",
    use_keypoints=False,
)

# 결과 디렉토리 찾기
result_dirs = sorted(Path("/home/gpu-agx/zoo/dataset/coco2017_val").glob("annotations*"), 
                     key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
if result_dirs:
    latest_dir_det = result_dirs[0]
    if latest_dir_det.name != "annotations2" and latest_dir_det.exists():
        # detection labels를 annotations2에 병합
        dest = Path("/home/gpu-agx/zoo/dataset/coco2017_val/annotations2")
        source_labels_dir = latest_dir_det / "labels"
        if source_labels_dir.exists():
            for subdir in source_labels_dir.iterdir():
                if subdir.is_dir():
                    dest_subdir = dest / "labels" / subdir.name
                    if dest_subdir.exists():
                        shutil.rmtree(str(dest_subdir))
                    shutil.copytree(str(subdir), str(dest_subdir))
                    print(f"Detection labels ({subdir.name})를 annotations2에 추가 완료")
                    print(f"추가된 label 파일 수: {len(list(dest_subdir.glob('*.txt')))}개")

# 임시 디렉토리 삭제
shutil.rmtree(str(temp_dir_det))
print("instances_val2017.json 변환 완료\n")

print("=" * 80)
print("변환 완료!")
print("=" * 80)
dest = Path("/home/gpu-agx/zoo/dataset/coco2017_val/annotations2")
if dest.exists():
    pose_count = len(list((dest / "labels").glob("**/*.txt"))) if (dest / "labels").exists() else 0
    print(f"결과 저장 위치: {dest}")
    print(f"총 label 파일 수: {pose_count}개")
