# 개선된 추론 시간 측정 스크립트

## 문제점 분석

기존 스크립트에서 `alone` 실행 시와 `together` 실행 시 측정 결과가 이상하게 나오는 현상의 원인:

1. **동기화 문제**: `torch.cuda.synchronize()`는 PyTorch의 CUDA 작업만 동기화하지만, TensorRT는 별도 스트림/컨텍스트로 동작할 수 있어서 실제 TRT inference가 끝나기 전에 측정이 끝날 수 있음
2. **DVFS/클럭 스케일링**: AGX의 전력 관리로 인해 부하에 따라 GPU 클럭이 변동
3. **엔드투엔드 측정**: 전처리/후처리 시간이 포함되어 CPU 오버헤드/캐시 효과가 결과에 영향

## 개선 사항

### 1. CUDA Runtime 동기화
- `torch.cuda.synchronize()` 대신 `cudaDeviceSynchronize()` 사용
- TensorRT 작업 완료까지 정확히 대기
- `common_sync.py` 모듈로 공통화

### 2. Tegrastats 모니터링
- GPU 클럭, 사용률, 전력 등을 실시간 로깅
- 각 스크립트별로 별도 로그 파일 생성
- DVFS 효과 확인 가능

### 3. 측정 정확도 향상
- 각 이미지 추론 전에 동기화 추가하여 이전 작업 완료 보장
- 측정 시작/종료 시점 명확화

## 사용 방법

### Alone 실행 (순차 실행)

```bash
cd /home/gpu-agx/zoo/script/InferenceTime_alone
python3 alone_sequence_improved.py
```

결과:
- `zoo/result/InferenceTime_alone/alone_sequence_improved.log`
- `zoo/result/InferenceTime_alone/alone_sequence_improved.txt`
- `zoo/result/InferenceTime_alone/tegrastats.log`

### Together 실행 (동시 실행)

1. 시작 신호 파일 삭제 (이전 실행 정리):
```bash
rm -f /home/gpu-agx/zoo/start/1
```

2. 5개 프로세스를 백그라운드로 동시 실행:
```bash
cd /home/gpu-agx/zoo/script/InferenceTime_together
python3 run_cls_improved.py &
python3 run_detect_improved.py &
python3 run_pose_improved.py &
python3 run_seg_improved.py &
python3 run_obb_improved.py &
```

3. 시작 신호 파일 생성 (모든 프로세스가 동시에 시작):
```bash
touch /home/gpu-agx/zoo/start/1
```

4. 모든 프로세스 완료 대기:
```bash
wait
```

결과:
- `zoo/result/InferenceTime_together/together_improved.log`
- `zoo/result/InferenceTime_together/together_improved.txt`
- `zoo/result/InferenceTime_together/tegrastats_*.log` (각 모델별)

## 결과 비교

개선된 버전으로 측정하면:
- **동시 실행 시 더 빠르게 나오는 현상이 사라지거나 줄어듦**
- **Tegrastats 로그로 GPU 클럭/사용률 확인 가능**
- **더 정확한 TensorRT inference 시간 측정**

## Tegrastats 로그 분석

tegrastats 로그에서 확인할 수 있는 정보:
- GPU 클럭 (GR3D freq)
- GPU 사용률 (GR3D %)
- 전력 소비
- CPU 클럭/사용률

예시:
```
RAM 1234/12345MB (lfb 1234x4MB) CPU [12%@1234,12%@1234,12%@1234,12%@1234] EMC_FREQ 0% GR3D_FREQ 0% APE 25 MTS fg 0% bgnd 0% ...
```

## 주의사항

1. **클럭 고정 테스트**: 더 정확한 비교를 위해 클럭을 고정할 수 있습니다:
```bash
sudo jetson_clocks  # 최대 클럭으로 고정
# 또는
sudo nvpmodel -m 0  # 최대 성능 모드
```

2. **Tegrastats 권한**: tegrastats는 root 권한이 필요할 수 있습니다. 필요시 스크립트를 sudo로 실행하거나 tegrastats 실행 부분을 수정하세요.

3. **결과 해석**: 
   - 개선된 버전에서도 동시 실행이 더 빠르게 나온다면, 실제로 GPU가 병목이 아니거나 다른 요인(캐시, 스케줄링 등)이 작용하는 것일 수 있습니다.
   - Tegrastats 로그를 확인하여 GPU 사용률이 100%에 가까운지 확인하세요.







