#!/bin/bash
# 로컬 컴퓨터에서 실행할 스크립트
# 이 스크립트를 로컬 컴퓨터로 복사해서 실행하세요

SERVER="gpu-agx@155.230.16.157"
PORT="40010"
REMOTE_PATH="/home/gpu-agx/zoo/result"
LOCAL_PATH="./result"

echo "=== 결과 파일 다운로드 ==="
echo "서버: $SERVER"
echo "포트: $PORT"
echo "원격 경로: $REMOTE_PATH"
echo "로컬 경로: $LOCAL_PATH"
echo ""

# 방법 선택
echo "다운로드 방법을 선택하세요:"
echo "1) SCP (간단)"
echo "2) RSYNC (효율적, 재전송 가능)"
echo "3) 압축 파일로 다운로드"
read -p "선택 (1-3): " choice

case $choice in
    1)
        echo "SCP로 다운로드 중..."
        scp -P $PORT -r $SERVER:$REMOTE_PATH $LOCAL_PATH
        ;;
    2)
        echo "RSYNC로 동기화 중..."
        rsync -avz -e "ssh -p $PORT" $SERVER:$REMOTE_PATH/ $LOCAL_PATH/
        ;;
    3)
        echo "서버에서 압축 중..."
        ssh -p $PORT $SERVER "cd /home/gpu-agx/zoo && tar -czf result.tar.gz result/"
        echo "압축 파일 다운로드 중..."
        scp -P $PORT $SERVER:/home/gpu-agx/zoo/result.tar.gz ./
        echo "압축 해제 중..."
        tar -xzf result.tar.gz
        echo "서버에서 압축 파일 삭제 중..."
        ssh -p $PORT $SERVER "rm /home/gpu-agx/zoo/result.tar.gz"
        ;;
    *)
        echo "잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "완료! 파일이 $LOCAL_PATH 에 다운로드되었습니다."


