# KICT-RAIN-AI_ver1
기법: 재귀적 학습전략이 포함된 U-net 기반의 모델로, inference에서 재귀적 예측을 사용하지 않더라도 4개의 입력 강우장으로  18개(10분~180분) 예측강우장 생성<br/>
사전학습된 예측강우 모델: model-best_rec_180min_f.h5 <br/>

# KICT-RAIN-AI_ver2
기법: 매 선행 예측시점별 U-net을 최적화, 1개의 모델이 1개의 예측강우장 생성<br/>
사전학습된 예측강우 모델: model-best_fcst_10min.h5~model-best_fcst_180min.h5, 18개<br/>
