# SCP 파일 전송 가이드

## 다운로드 (서버 → 로컬)
```bash
scp -P 40010 -r gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result ./result
```

## 업로드 (로컬 → 서버)
```bash
scp -P 40010 -r ./result gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result
```

## 차이점
- **다운로드**: `서버경로 로컬경로`
- **업로드**: `로컬경로 서버경로`

## 상세 예시

### 1. 전체 폴더 업로드
```bash
# 로컬의 result 폴더를 서버로 업로드
scp -P 40010 -r ./result gpu-agx@155.230.16.157:/home/gpu-agx/zoo/
```

### 2. 특정 파일만 업로드
```bash
# 단일 파일
scp -P 40010 ./file.txt gpu-agx@155.230.16.157:/home/gpu-agx/zoo/

# 여러 파일
scp -P 40010 ./file1.txt ./file2.txt gpu-agx@155.230.16.157:/home/gpu-agx/zoo/
```

### 3. 압축 파일 업로드
```bash
# 압축 파일 업로드
scp -P 40010 ./result.tar.gz gpu-agx@155.230.16.157:/home/gpu-agx/zoo/

# 서버에서 압축 해제
ssh -p 40010 gpu-agx@155.230.16.157 "cd /home/gpu-agx/zoo && tar -xzf result.tar.gz"
```

## 옵션 설명
- `-P 40010`: 포트 번호 지정 (기본 22가 아닐 때)
- `-r`: 재귀적으로 폴더 전체 복사
- `-v`: 상세한 진행 상황 출력 (verbose)
- `-p`: 파일의 수정 시간, 권한 등 보존

## RSYNC 사용 (더 효율적)
```bash
# 업로드
rsync -avz -e "ssh -p 40010" ./result/ gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result/

# 다운로드
rsync -avz -e "ssh -p 40010" gpu-agx@155.230.16.157:/home/gpu-agx/zoo/result/ ./result/
```

## 서버 정보
- 호스트: `155.230.16.157`
- 포트: `40010`
- 사용자: `gpu-agx`




