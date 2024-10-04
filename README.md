# pyEgg
달걀 공장에서 달걀을 낳지 않는 닭이 사료를 소비하는 문제를 해결하기 위해, 케이지별 산란수를 측정하여 달걀을 낳지 않는 닭이 위치한 케이지를 선별하는 알고리즘입니다.

![pyEgg](https://github.com/user-attachments/assets/edaf16e7-cbc2-4e8e-9571-ce6dabb26212)

알고리즘은 다음과 같습니다.
-	첫째, 매번 달라지는 컨베이어 벨트의 속도를 측정하기 위해 벨트에 빨간색 테이프 2개를 1m 간격으로 설치하고, 첫번째 테이프 검출 시점으로부터 두번째 테이프 검출 시점까지 시간을 측정하여 ‘컨베이어 벨트의 속도’를 측정
-	둘째, ‘컨베이어 벨트의 속도’와 ‘달걀 검출까지 걸린 시간’을 이용하여 ‘카메라로부터 달걀까지의 거리’ 측정
-	셋째, ‘카메라로부터 달걀까지의 거리’를 이용하여 달걀이 검출된 케이지 번호를 특정하고, 산란수가 평균보다 적은 케이지를 달걀을 낳지 않는 닭이 존재하는 케이지로 선별


## :heavy_check_mark: Tested

| Python | pytorch |  Windows   |   Mac   |   Linux  |
| :----: | :-----: | :--------: | :-----: | :------: |
| 3.8.0+ | 1.10.0+ | Windows 10 | X |  Ubuntu 18.04 |


## :arrow_down: Installation

```bash
conda create -n pyEgg python=3.8.0
conda activate pyEgg
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


## :rocket: Getting started

You can inference with your own live camera stream 
or you can also inference with your own custom Video file in /pyEgg/src/input folder.
```bash
cd src
python main.py
```

## :clipboard: Reference
- https://github.com/RizwanMunawar/yolov7-object-tracking
- https://github.com/mikel-brostrom/Yolov7_DeepSort_Pytorch
- https://github.com/WongKinYiu/yolov7
