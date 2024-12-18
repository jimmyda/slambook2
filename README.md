# slambook2 practice

코드 설명
U-Net 모델 정의:

간단한 U-Net 구조를 구현했습니다.
encoder는 컨볼루션과 MaxPooling으로 다운샘플링하고, decoder는 업샘플링을 수행합니다.
데이터 로드:

SegmentationDataset 클래스는 이미지를 로드하고 정규화합니다.
이미지 및 마스크는 .png 형식으로 로드하여 토치 텐서로 변환합니다.
학습 루프:

손실 함수로 BCE(Binary Cross Entropy)를 사용합니다.
Adam Optimizer로 모델을 학습합니다.
데이터 경로:

data_dir 변수에 데이터셋의 경로를 지정하세요. 해당 경로에는 images와 masks 디렉토리가 있어야 합니다.
모델 저장:

학습 완료 후 모델을 unet_model.pth로 저장합니다.
