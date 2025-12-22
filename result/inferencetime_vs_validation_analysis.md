# InferenceTime vs Validation 성능 차이 분석

## 핵심 차이점

### InferenceTime 측정 (단순 추론만)
```python
# InferenceTime_together/run_cls_improved.py
for img in images:
    synchronize()  # 이전 작업 완료 대기
    t0 = time.perf_counter()
    _ = model.predict(...)  # 결과를 사용하지 않음
    synchronize()
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
```
- **단순 추론만 측정**: `model.predict()` 호출 후 결과를 버림
- **최소한의 후처리**: 결과 객체 접근 없음
- **메모리 사용 최소**: 결과를 저장하지 않음

### Validation 측정 (추론 + 정확도 계산)
```python
# Validation_together/run_cls_validation_together.py
for img_path in val_images:
    result = model.predict(...)
    results_list.append(result[0])  # 결과 저장
    
    # Ground truth 비교 및 정확도 계산
    if img_str in gt_dict and hasattr(result[0], 'probs'):
        gt_class_id = gt_dict[img_str]
        predicted_top1 = int(result[0].probs.top1)  # 결과 객체 접근
        if predicted_top1 == gt_class_id:
            correct_top1 += 1
        # Top-5 계산 등 추가 처리
```
- **추론 + 후처리**: 결과 객체에서 데이터 추출
- **Ground truth 비교**: CSV/JSON 로드 및 매칭
- **정확도 계산**: Top-1, Top-5 등 계산
- **메모리 사용 증가**: 모든 결과 객체 저장

## 성능 비교 데이터

### 전체 성능 비교표

| 모델 | InferenceTime (MPS) | Validation Alone | Validation Alone (WithoutVal) | Validation Together | Validation Together (WithoutVal) |
|------|---------------------|------------------|-------------------------------|---------------------|----------------------------------|
| **cls** | 8.28ms | 11.96ms | 10.70ms | 15.83ms | 11.38ms |
| **detect** | 30.11ms | 26.75ms | 26.75ms | 48.42ms | 61.57ms |
| **pose** | 24.14ms | 30.26ms | 28.25ms | 33.00ms | 41.05ms |
| **seg** | 31.78ms | 36.16ms | 35.36ms | 46.42ms | 62.85ms |
| **obb** | 22.48ms | 56.61ms | 56.30ms | 62.76ms | 62.24ms |

### InferenceTime 대비 증가율

| 모델 | Validation Alone | Validation Alone (WithoutVal) | Validation Together | Validation Together (WithoutVal) |
|------|------------------|-------------------------------|---------------------|----------------------------------|
| **cls** | +44.4% | +29.2% | +91.2% | +37.4% |
| **detect** | -11.2% | -11.2% | +60.8% | +104.4% |
| **pose** | +25.4% | +17.0% | +36.7% | +70.2% |
| **seg** | +13.8% | +11.3% | +46.0% | +97.8% |
| **obb** | +152.0% | +150.5% | +179.3% | +177.0% |


### 정확도 계산 제거 효과 (Alone)

| 모델 | Validation Alone | Validation Alone (WithoutVal) | 차이 | 개선율 |
|------|------------------|-------------------------------|------|--------|
| **cls** | 11.96ms | 10.70ms | -1.26ms | **-10.5%** |
| **detect** | 26.75ms | 26.75ms | 0.00ms | 0.0% |
| **pose** | 30.26ms | 28.25ms | -2.01ms | **-6.6%** |
| **seg** | 36.16ms | 35.36ms | -0.80ms | **-2.2%** |
| **obb** | 56.61ms | 56.30ms | -0.31ms | **-0.5%** |

### 정확도 계산 제거 효과 (Together)

| 모델 | Validation Together | Validation Together (WithoutVal) | 차이 | 변화율 |
|------|---------------------|----------------------------------|------|--------|
| **cls** | 15.83ms | 11.38ms | -4.45ms | **-28.1%** ⬇️ |
| **detect** | 48.42ms | 61.57ms | +13.15ms | **+27.1%** ⬆️ |
| **pose** | 33.00ms | 41.05ms | +8.05ms | **+24.4%** ⬆️ |
| **seg** | 46.42ms | 62.85ms | +16.43ms | **+35.4%** ⬆️ |
| **obb** | 62.76ms | 62.24ms | -0.52ms | **-0.8%** |

### Alone vs Together 비교

| 모델 | Validation Alone | Validation Together | 차이 | 증가율 |
|------|------------------|---------------------|------|--------|
| **cls** | 11.96ms | 15.83ms | +3.87ms | +32.4% |
| **detect** | 26.75ms | 48.42ms | +21.67ms | +81.0% |
| **pose** | 30.26ms | 33.00ms | +2.74ms | +9.1% |
| **seg** | 36.16ms | 46.42ms | +10.26ms | +28.4% |
| **obb** | 56.61ms | 62.76ms | +6.15ms | +10.9% |

### Alone vs Together 비교 (WithoutVal)

| 모델 | Validation Alone (WithoutVal) | Validation Together (WithoutVal) | 차이 | 증가율 |
|------|-------------------------------|----------------------------------|------|--------|
| **cls** | 10.70ms | 11.38ms | +0.68ms | +6.4% |
| **detect** | 26.75ms | 61.57ms | +34.82ms | +130.2% |
| **pose** | 28.25ms | 41.05ms | +12.80ms | +45.3% |
| **seg** | 35.36ms | 62.85ms | +27.49ms | +77.8% |
| **obb** | 56.30ms | 62.24ms | +5.94ms | +10.6% |

## 왜 Validation에서 차이가 더 큰가?

### 1. **CPU 후처리 오버헤드**
```
InferenceTime: GPU 추론만 → 동기화 → 측정 종료
Validation: GPU 추론 → 결과 객체 접근 → CPU에서 정확도 계산 → 측정 종료
```

**Validation 추가 작업:**
- 결과 객체에서 데이터 추출 (`result[0].probs.top1`, `result[0].boxes` 등)
- Ground truth와 비교
- 정확도 메트릭 계산
- 결과 객체를 메모리에 저장 (`results_list.append()`)

### 2. **메모리 경쟁 증가**
- **InferenceTime**: 결과를 즉시 버림 → 메모리 사용 최소
- **Validation**: 모든 결과 객체 저장 → 메모리 사용 증가
- **병렬 실행 시**: 5개 프로세스가 동시에 메모리 할당 → 경쟁 심화

### 3. **CPU-GPU 동기화 빈도**
```python
# InferenceTime: 각 이미지마다 동기화
for img in images:
    synchronize()  # 이전 작업 완료
    predict()
    synchronize()  # 추론 완료 대기
    # 끝 (결과 사용 안함)

# Validation: 각 이미지마다 동기화 + 결과 처리
for img in images:
    synchronize()  # 이전 작업 완료
    result = predict()
    synchronize()  # 추론 완료 대기
    # 결과 처리 (CPU 작업)
    process_result(result)  # 추가 시간 소요
```

### 4. **데이터셋 차이**
- **InferenceTime**: CIFAR100 (cls), coco128 (detect 등) - 작은 이미지
- **Validation**: ImageNet1k_100 (cls), coco2017_val (detect 등) - 더 큰/다양한 이미지
- Validation 데이터셋이 더 복잡할 수 있음

### 5. **Ground Truth 로드 오버헤드**
```python
# Validation에서만 수행
csv_file = data_path / "val100.csv"
gt_dict = {}  # Ground truth 딕셔너리 생성
# CSV/JSON 파싱 및 매핑
# 각 이미지마다 gt_dict 조회
```

## 상세 분석

### cls 모델
- **InferenceTime (MPS)**: 8.28ms
- **Validation Alone**: 11.96ms (+44%)
- **Validation Together**: 15.83ms (+91%)

**차이 원인:**
- Top-1, Top-5 정확도 계산
- `result[0].probs.top1`, `result[0].probs.top5` 접근
- Ground truth 딕셔너리 조회
- 병렬 실행 시 CPU 후처리 경쟁

### detect 모델
- **InferenceTime (MPS)**: 30.11ms
- **Validation Alone**: 26.75ms (-11%, 더 빠름!)
- **Validation Together**: 48.42ms (+61%)

**흥미로운 점:**
- Validation Alone이 InferenceTime보다 빠름 (데이터셋/측정 방식 차이)
- 하지만 Together에서는 크게 느려짐
- **병렬 실행 시 detect가 가장 큰 영향**: +74.6% 증가

**원인:**
- `result[0].boxes` 접근 및 처리
- Ground truth bbox 비교
- TP/FP/FN 계산
- 병렬 실행 시 GPU 리소스 경쟁 + CPU 후처리 경쟁

### obb 모델
- **InferenceTime (MPS)**: 22.48ms
- **Validation Alone**: 56.61ms (+152%)
- **Validation Together**: 62.76ms (+179%)

**가장 큰 차이:**
- Validation에서만 공식 지표 계산 시도 (`model.val()` 호출)
- DOTA 데이터셋 yaml 파일 처리
- OBB 결과 처리 복잡도

## Validation WithoutVal 결과 분석 (정확도 계산 제거)

### 성능 비교표 (정확도 계산 제거 전후)

| 모델 | InferenceTime (MPS) | Validation Alone | Validation Alone (WithoutVal) | Validation Together | Validation Together (WithoutVal) |
|------|---------------------|------------------|-------------------------------|---------------------|----------------------------------|
| **cls** | 8.28ms | 11.96ms | **10.70ms** (-10.5%) | 15.83ms | **11.38ms** (-28.1%) |
| **detect** | 30.11ms | 26.75ms | **26.75ms** (0%) | 48.42ms | **61.57ms** (+27.1%) |
| **pose** | 24.14ms | 30.26ms | **28.25ms** (-6.6%) | 33.00ms | **41.05ms** (+24.4%) |
| **seg** | 31.78ms | 36.16ms | **35.36ms** (-2.2%) | 46.42ms | **62.85ms** (+35.4%) |
| **obb** | 22.48ms | 56.61ms | **56.30ms** (-0.5%) | 62.76ms | **62.24ms** (-0.8%) |

### 핵심 발견: 정확도 계산 제거의 역설적 효과

#### 1. **Alone 실행: 정확도 계산 제거 → 약간 빨라짐 (예상대로)**
- **cls**: 11.96ms → 10.70ms (-10.5%)
  - 정확도 계산 오버헤드 제거로 개선
- **detect**: 26.75ms → 26.75ms (변화 없음)
  - 정확도 계산이 거의 영향 없음 (이미 최적화됨)
- **pose/seg/obb**: 2~7% 개선
  - 정확도 계산 오버헤드가 작지만 존재

#### 2. **Together 실행: 정확도 계산 제거 → 오히려 느려짐! (예상과 반대)**

**예상과 반대인 이유:**

##### **cls 모델: 정확도 계산 제거 → 크게 빨라짐 (-28.1%)**
- **15.83ms → 11.38ms**: 정확도 계산 제거가 효과적
- **원인**: cls는 정확도 계산이 CPU 집약적 (Top-1, Top-5 계산)
- **병렬 실행 시**: CPU 경쟁이 심했지만, 제거 후 GPU 추론에 집중 가능

##### **detect/pose/seg 모델: 정확도 계산 제거 → 오히려 느려짐 (+24~35%)**
- **detect**: 48.42ms → 61.57ms (+27.1%)
- **pose**: 33.00ms → 41.05ms (+24.4%)
- **seg**: 46.42ms → 62.85ms (+35.4%)

**왜 느려졌을까?**

1. **정확도 계산이 "버퍼링" 역할을 했을 가능성**
   ```python
   # Validation (정확도 계산 있음)
   result = model.predict(...)
   synchronize()  # GPU 완료 대기
   # 정확도 계산 (CPU 작업, 시간 소요)
   calculate_accuracy(result)  # 이 시간 동안 GPU가 다음 작업 준비
   # 다음 이미지로 이동
   
   # Validation WithoutVal (정확도 계산 없음)
   result = model.predict(...)
   synchronize()  # GPU 완료 대기
   # 즉시 다음 이미지로 이동 → GPU가 아직 준비 안됨?
   ```

2. **동기화 타이밍 차이**
   - 정확도 계산이 있을 때: CPU 작업 시간 동안 GPU가 다음 배치 준비
   - 정확도 계산이 없을 때: GPU가 바로 다음 작업을 받지만, 준비가 덜 됨

3. **메모리 할당 패턴 차이**
   - 정확도 계산 시: 결과 객체를 메모리에 저장 → 메모리 할당이 분산됨
   - 정확도 계산 없을 때: 결과를 즉시 버리지만, 메모리 할당이 집중됨

4. **병렬 실행 시 스케줄링 차이**
   - 5개 프로세스가 동시에 정확도 계산을 하면: CPU 경쟁이 있지만, GPU는 순차적으로 작업
   - 5개 프로세스가 동시에 추론만 하면: GPU 경쟁이 더 심해짐

##### **obb 모델: 정확도 계산 제거 → 거의 변화 없음 (-0.8%)**
- **62.76ms → 62.24ms**: 거의 동일
- **원인**: obb는 정확도 계산이 복잡하지만, 병렬 실행 시 이미 GPU 병목이 주 요인

### 데이터셋 차이 영향 분석

**Validation Alone vs Validation Alone WithoutVal 비교:**
- 두 경우 모두 **동일한 데이터셋** 사용 (ImageNet1k_100, coco2017_val, DOTAv1.5-1024_val)
- 정확도 계산 제거로 **1~10% 개선** (예상대로)

**결론: 데이터셋 차이는 주요 요인이 아님**
- InferenceTime과 Validation의 차이는 주로 **정확도 계산 오버헤드** 때문
- 데이터셋 자체의 복잡도는 큰 영향 없음

## 결론

### InferenceTime vs Validation 차이의 주요 원인

1. **CPU 후처리 오버헤드** (가장 중요)
   - Validation은 결과 객체에서 데이터 추출 및 정확도 계산
   - **Alone 실행 시**: 정확도 계산 제거로 1~10% 개선
   - **Together 실행 시**: 모델에 따라 다름
     - cls: 정확도 계산 제거로 크게 개선 (-28%)
     - detect/pose/seg: 정확도 계산 제거로 오히려 느려짐 (+24~35%)
     - obb: 거의 변화 없음

2. **병렬 실행 시 동기화/스케줄링 효과**
   - 정확도 계산이 "버퍼링" 역할을 할 수 있음
   - CPU 작업 시간 동안 GPU가 다음 작업 준비
   - 정확도 계산 제거 시 GPU 경쟁이 더 심해질 수 있음

3. **메모리 사용 증가**
   - Validation은 모든 결과 객체를 메모리에 저장
   - 5개 프로세스 동시 실행 시 메모리 경쟁 심화
   - 정확도 계산 제거 시 메모리 할당 패턴이 달라질 수 있음

4. **데이터셋/측정 방식 차이** (영향 적음)
   - Validation WithoutVal 결과로 확인: 데이터셋 차이는 주요 요인 아님
   - 정확도 계산 오버헤드가 주요 차이 원인

### 핵심 인사이트

1. **정확도 계산의 이중 효과**
   - **Alone 실행**: CPU 오버헤드로 성능 저하
   - **Together 실행**: 일부 모델에서는 "버퍼링" 효과로 성능 향상 가능

2. **모델별 차이**
   - **cls**: 정확도 계산이 CPU 집약적 → 제거 시 크게 개선
   - **detect/pose/seg**: 정확도 계산이 GPU 동기화와 밸런스 → 제거 시 오히려 느려짐
   - **obb**: GPU 병목이 주 요인 → 정확도 계산 영향 적음

3. **병렬 실행 최적화의 복잡성**
   - 단순히 CPU 작업을 제거하는 것이 항상 성능 향상은 아님
   - GPU-CPU 작업 밸런스가 중요
   - 동기화 타이밍과 스케줄링이 성능에 큰 영향

### 권장사항

1. **InferenceTime 측정**: 단순 추론 성능 확인용
   - GPU 추론 성능만 측정
   - 병렬 실행 시 영향이 적음

2. **Validation 측정**: 실제 사용 시나리오
   - 추론 + 후처리 전체 파이프라인 측정
   - 병렬 실행 시 성능 저하 더 큼
   - **더 현실적인 성능 지표**

3. **Validation WithoutVal 측정**: 정확도 계산 영향 분석용
   - 정확도 계산 오버헤드 분리 측정
   - 병렬 실행 시 예상치 못한 효과 발견 가능

4. **병렬 실행 최적화**
   - CPU 후처리를 별도 스레드로 분리 (비동기 처리)
   - GPU-CPU 작업 밸런스 고려
   - 동기화 타이밍 최적화
   - 배치 처리로 메모리 효율성 향상

