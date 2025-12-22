


ssh gpu-agx@155.230.16.157 -p 40010


docker run --rm -it ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash

이미지 목록
docker images

컨테이너 목록
docker ps

컨테이너 실행
docker run -d --name container1 \
  -v /tmp/nvidia-mps:/tmp/nvidia-mps \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  /bin/bash -c "sleep infinity"


방법 2: 컨테이너 생성 시 바로 명령어 실행 (더 효율적)
docker run -d --name container5 \
  -v /tmp/nvidia-mps:/tmp/nvidia-mps \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
  -v /home/gpu-agx/zoo:/ultralytics/zoo \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  /bin/bash -c "python /ultralytics/zoo/script/alone_sequence.py"

  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \ 이걸로 CUDA에게 위치 알려주는역할을 한다.



docker exec -it container4 /bin/bash


results = model.predict(
    source="/ultralytics/shared/dataset/sample100",
    imgsz=224,
    device=0,
    save=False,
    verbose=False
)
현재 CLS만 거의 결과 값이 0이다. 


DOTA 100장 , ImageNet 100장있다 둘다 Val가능

DOTA100: images 100장, labels 100개로 카운트 일치 확인.
DOTA100 메타(metadata.json) 기준 클래스 10개, 총 100장(Train 80 / Val 20), 포맷 DOTA OBB(cx cy w h angle class_id difficulty).

ImageNet100: 100클래스 샘플(Train 80 / Val 20, 총 100장)로, 메타에도 100개 클래스가 번호별로 기재. 일반적인 ImageNet Val 점수 확인에는 클래스 수·데이터가 너무 작아 정식 검증 세트 대용으로는 부적절.




alone_sequence.py
/home/gpu-agx/zoo/script/alone_sequence.py에 5개 엔진(클래스/디텍션/포즈/세그/OBB)을 순차 실행해 이미지별 추론 시간과 평균을 측정하고 zoo/result/alone_sequence/alone_sequence.log, alone_sequence.txt에 기록하도록 추가했습니다. 데이터셋은 CLS→ImageNet100/train 100장, detect/pose/seg→coco128/images/train2017 128장, OBB→DOTA100/images 100장 사용하며, 첫 장은 웜업으로 측정에서 제외합니다. GPU 동기화(torch 유무 체크)로 순수 추론 시간만 집계합니다.


sequenc_alone 실행해서 순차 평균시간 기록
docker run -d --name alone \
  -v /tmp/nvidia-mps:/tmp/nvidia-mps \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
  -v /home/gpu-agx/zoo:/ultralytics/zoo \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  /bin/bash -c "python /ultralytics/zoo/script/alone_sequence.py"


# 실행/중지 모두 포함 모든 컨테이너 제거
docker rm -f $(docker ps -aq)


## alone_sequence : 하면 log, txt 생성해줌 각추론시간 남기고 마지막 추론은 뺀다. 
docker run -d --name alone --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_alone/alone_sequence.py"




# InferenceTime_together: 컨테이너 5개(MPS) 기동 후 start/1 생성 시 동시 시작
docker run -d --name cls --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_cls.py"
docker run -d --name detect --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_detect.py"
docker run -d --name pose --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_pose.py"
docker run -d --name seg --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_seg.py"
docker run -d --name obb --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_obb.py"


# Validation_alone
docker run -d --name alone --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/Validation_alone/validation_alone.py"

docker rm -f $(docker ps -aq)



현재까지 진행상황
공식지표 책정 실패 


cd /mnt/sd



다다운로드 (서버 → 로컬): 서버경로 로컬경로
  scp -P 40010 -r gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result ./result

업로드 (로컬 → 서버): 로컬경로 서버경로
  scp -P 40010 -r ./result gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result





# improve alone 실행 
docker run -d --name alone --runtime=nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_alone/alone_sequence_improved.py"


docker rm -f $(docker ps -aq)
# InferenceTime_together_improved : 컨테이너 5개(MPS) 기동 후 start/1 생성 시 동시 시작
docker run -d --name cls --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_cls_improved.py"
docker run -d --name detect --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_detect_improved.py"
docker run -d --name pose --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_pose_improved.py"
docker run -d --name seg --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_seg_improved.py"
docker run -d --name obb --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_obb_improved.py"

docker rm -f $(docker ps -aq)
# InferenceTime_together_improved (MPS 없이) : 컨테이너 5개 기동 후 start/1 생성 시 동시 시작
docker run -d --name cls --runtime nvidia -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_cls_improved.py"
docker run -d --name detect --runtime nvidia -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_detect_improved.py"
docker run -d --name pose --runtime nvidia -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_pose_improved.py"
docker run -d --name seg --runtime nvidia -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_seg_improved.py"
docker run -d --name obb --runtime nvidia -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/InferenceTime_together/run_obb_improved.py"




이거쓰면 각 컨테이너의 사용량 알 수 있다. 
docker stats cls detect pose seg obb 

CPU %: CPU 사용률
MEM USAGE / LIMIT: 메모리 사용량 / 제한
MEM %: 메모리 사용률
NET I/O: 네트워크 입출력
BLOCK I/O: 디스크 입출력
PIDS: 프로세스 수


1218 현재상황 , 알맞는 데이텃세 다운로드중, 200개로 출여서 올것이다. 
추론 같은경우는 진척이 꽤있었다. 이제 이게 어디서 병목 발생하는직 중요할것 같다.

카톡으로 말할 것. 성능에는 거의 차이가 없었다. MPS에서 얼마나 자원들이 잘 아름답게 사용되고 있는지 
not using MPS랑 비교해보면 될까? 

1차적으로 MPS에서는 병목이 발생하지 않는다는 결론 , MPS 안쓸때 대비 어떤 부분에서 잘 처리를 해주고 있을까 , 그리고 나머지는 느는데 CLS에 걸리는 시간은 가장 줄까 ?? 이게 신기하다 


PPT만들때 정확히 어떻게 추론시간만 얻어냈는지 확인하기 



# 한 번만 확인 (현재 상태)
docker stats --no-stream cls detect pose seg obb

# 실시간 모니터링 (계속 업데이트)
docker stats cls detect pose seg obb


docker stats --no-stream cls detect pose seg obb > docker_stats.log



# 특정 컨테이너만
docker stats --no-stream cls


# GPU 사용률 모니터링
nvidia-smi -l 1

# 또는 tegrastats (Jetson 전용)
tegrastats --interval 100 > /home/gpu-agx/zoo/result/tegrastats.log



tegrastats --interval 100 > /home/gpu-agx/zoo/result/tegrastats.log 이거하는거랑  # 실시간 모니터링 (계속 업데이트)
docker stats cls detect pose seg obb 이거하는거랑 뭐가 더 오버헤드가 적을까 ? docker stats cls detect pose seg obb이게 도커의 테그라 스텟 이런식으로 봐도될까? tegrastats --interval 100 > /home/gpu-agx/zoo/result/tegrastats.log 이걸쓴다면 도커 돌리면서 밖에서 돌리고있어야겠지  vmstat이랑 이것들이랑은 뭐가 다르고 vmstat도 실시간으로 로그남길수있나 이것들다 오버헤드들은 비슷한가? nsite는 또 뭐가달라. 각각 오버헤드들과 로그들을 남길수 있는지 , 도커내부에서 실행가능한지 궁금해. 각 강점이뭔지


CONDA
eval "$(/mnt/sd/gpu-agx/anaconda3/bin/conda shell.bash hook)"
conda activate base

conda create -n yolo python=3.10 -y
conda activate yolo && pip install -r /home/gpu-agx/zoo/dataset/JSON2YOLO/requirements.txt


docker rm -f $(docker ps -aq)

docker run -d --name alone --runtime nvidia -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -v /home/gpu-agx/zoo:/ultralytics/zoo ultralytics/ultralytics:latest-jetson-jetpack6 /bin/bash -c "python /ultralytics/zoo/script/Validation_objectdetect/validate_pose.py"