# KICT-RAIN-AI
KICT AI based RAIN prediction model
# 모델 설명
기상청/환경부 레이더 격자 강우, 혹은 강우분포 정보를 이용하여 선행 10분에서 180분까지 예측강우생성하는 딥러닝 U-Net 기반 강우예측모델
# 모델 실행
실행환경: Google Colab  <br/>
실행코드:Inference_KICT_RAIN_AI_525x625_github.ipynb <br/>
실행코드(2025.10.02):Inference_KICT_RAIN_AI_v3_525x625_github.ipynb <br/> 
내용: 사전학습된 모델('.h5','.tflite') 이용하여 예측 수행 <br/>

사전학습모델 1: model-best_rec_180min_f.h5 재귀적 학습 네트워크를 이용하여 사전학습된 예측모델 <br/>
(참고: 윤성심(2022) 초단시간 강우예측을 위한 U-Net 기반 재귀적 예측 모델, 디지털콘텐츠학회논문지, Vol.23, No.12,pp.2481-2488) <br/>

사전학습모델 2: model-best_fcst_10min.tflite~model-best_fcst_180min.tflite 각 예측 선행 시점별 최적학습된 예측모델 <br/>

사전학습모델 3: model-best_fcst_10min_re.tflite~model-best_fcst_180min_re.tflite,각 예측 선행 시점별 최적학습된 예측모델<br/>
              (갱신정보: 기상청 레이더 관측강우 자료를 활용한 KICT-RAIN-AI_ver2 모델 전이학습 수행결과) <br/>


