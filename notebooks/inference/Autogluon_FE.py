# %% [라이브러리 임포트 및 경고 설정]
import pandas as pd
import numpy as np
import warnings
from sklearn.utils.class_weight import compute_class_weight
from autogluon.multimodal import MultiModalPredictor

warnings.filterwarnings('ignore')

# %% [데이터 로드]
# 데이터를 CSV 파일에서 읽어옵니다.
try:
    final_train = pd.read_csv('../../data/preprocessed_data/final_train.csv')
    final_test = pd.read_csv('../../data/preprocessed_data/final_test.csv')
    print("데이터 로드 성공.")
except Exception as e:
    print("데이터 로드 에러:", e)
    final_train = pd.DataFrame()
    final_test = pd.DataFrame()

# %% [컬럼 타입 설정]
# 각 컬럼의 타입을 지정합니다.
try:
    column_types = {
       'URL': 'text',
       'label': 'categorical',
       'digit_ratio': 'numerical',
       'special_char_count': 'numerical',
       'subdomain_count': 'numerical',
       'length': 'numerical'
    }
    print("컬럼 타입 설정 성공.")
except Exception as e:
    print("컬럼 타입 설정 에러:", e)
    column_types = {}

# %% [클래스 가중치 계산]
# train 데이터의 'label' 컬럼을 기준으로 클래스 가중치를 계산하고 정규화합니다.
try:
    weights = compute_class_weight(
       class_weight='balanced',
       classes=np.unique(final_train['label']),
       y=final_train['label'].values
    )
    weights = weights / weights.sum()  # 가중치 정규화 (합계 1)
    weights = list(weights)
    print("\n계산된 클래스 가중치:", weights)
except Exception as e:
    print("클래스 가중치 계산 에러:", e)
    weights = []

# %% [MultiModalPredictor 생성]
# AutoGluon MultiModalPredictor를 생성합니다.
try:
    predictor = MultiModalPredictor(
       label='label',
       problem_type='binary',
       eval_metric='roc_auc',
       validation_metric='roc_auc'
    )
    print("Predictor 생성 성공.")
except Exception as e:
    print("MultiModalPredictor 생성 에러:", e)
    predictor = None

# %% [모델 학습]
# train 데이터를 사용하여 모델을 학습시킵니다.
try:
    if predictor is not None:
        predictor.fit(
           train_data=final_train,
           column_types=column_types,
           presets='best_quality',
           time_limit=None,
           seed=42,
           hyperparameters={
              "model.hf_text.checkpoint_name": "r3ddkahili/final-complete-malicious-url-model",
              "env.per_gpu_batch_size": 64,
              "optimization.patience": 3,
              "optimization.loss_function": "focal_loss",
              "optimization.focal_loss.alpha": weights,
           }
        )
        print("모델 학습 완료.")
    else:
        print("Predictor가 None입니다. 모델 학습 건너뜁니다.")
except Exception as e:
    print("predictor.fit 실행 중 에러:", e)

# %% [모델 예측]
# 학습된 모델을 사용하여 test 데이터에 대한 예측 확률을 계산합니다.
try:
    if predictor is not None:
        test_pred_proba = predictor.predict_proba(final_test)
        # 이진 분류의 경우, 클래스 1(악성 URL)의 확률 사용
        if 1 in test_pred_proba.columns:
            prediction_scores = test_pred_proba[1]
        else:
            prediction_scores = test_pred_proba.iloc[:, 1]
        print("예측 완료.")
    else:
        print("Predictor가 None입니다. 기본 0 예측값 사용.")
        prediction_scores = np.zeros(len(final_test))
except Exception as e:
    print("예측 실행 중 에러:", e)
    prediction_scores = np.zeros(len(final_test))

# %% [제출 파일 생성]
# 예측 결과를 기반으로 제출 파일을 생성합니다.
try:
    submission = pd.DataFrame({
       'ID': final_test['ID'],
       'probability': prediction_scores
    })
    submission.to_csv('./submission/FE_multimodal.csv', index=False)
    print("\n제출 파일 생성 완료.")
except Exception as e:
    print("제출 파일 생성 에러:", e)
