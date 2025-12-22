# 결과 파일 다운로드 가이드

## 방법 1: SCP 사용 (가장 간단)

### 로컬 컴퓨터에서 실행:
```bash
# 전체 result 폴더 다운로드
scp -r gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result ./result

# 또는 특정 폴더만
scp -r gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result/Validation_alone ./Validation_alone
scp -r gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result/InferenceTime_alone ./InferenceTime_alone
scp -r gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result/InferenceTime_together ./InferenceTime_together
```

### 포트가 다르면 (40010):
```bash
scp -P 40010 -r gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result ./result
```

## 방법 2: RSYNC 사용 (더 효율적, 재전송 가능)

### 로컬 컴퓨터에서 실행:
```bash
# 전체 동기화
rsync -avz -e "ssh -p 40010" gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result/ ./result/

# 특정 폴더만
rsync -avz -e "ssh -p 40010" gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result/Validation_alone/ ./Validation_alone/
```

## 방법 3: 압축 후 전송 (대용량일 때)

### 서버에서 실행:
```bash
cd /home/gpu-agx/zoo
tar -czf result.tar.gz result/
```

### 로컬 컴퓨터에서 실행:
```bash
scp -P 40010 gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result.tar.gz ./
tar -xzf result.tar.gz
```

## 방법 4: SFTP 사용 (대화형)

### 로컬 컴퓨터에서 실행:
```bash
sftp -P 40010 gpu-agx@155.230.16.157
# SFTP 프롬프트에서:
cd /home/gpu-agx/zoo/result
get -r Validation_alone
get -r InferenceTime_alone
get -r InferenceTime_together
exit
```

## 현재 파일 크기
- InferenceTime_alone: 148KB
- InferenceTime_together: 216KB
- Validation_alone: 116KB
- Validation_together: 4KB
- 총 약 500KB (48개 파일)

## 서버 정보
- 호스트: 155.230.16.157
- 포트: 40010 (기본 22가 아닐 수 있음)
- 사용자: gpu-agx
- 경로: /home/gpu-agx/zoo/result




