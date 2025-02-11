#%% md
# ## 1. Import

#%%
import pandas as pd
import numpy as np
import os
import re
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from autogluon.multimodal import MultiModalPredictor

warnings.filterwarnings('ignore')

#%% md
# ## 2. Data Load and Preprocessing

#%%
# 학습/평가 데이터 로드
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# '[.]'을 '.'으로 복구 (정규식 적용)
train_df['URL'] = train_df['URL'].str.replace(r'\[\.\]', '.', regex=True)
test_df['URL'] = test_df['URL'].str.replace(r'\[\.\]', '.', regex=True)

# URL 컬럼이 문자열인지 명시적으로 변환
train_df['URL'] = train_df['URL'].astype(str)
test_df['URL'] = test_df['URL'].astype(str)

#%% md
# ## 3. Compute Class Weights for Focal Loss

#%%
# train_df의 'label' 컬럼을 기준으로 balanced 가중치 계산
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label'].values
)
# 가중치를 정규화 (합이 1이 되도록)
weights = weights / weights.sum()
weights = list(weights)
print("Computed class weights:", weights)

#%% md
# ## 4. Model Training using MultiModalPredictor

#%%
# MultiModalPredictor를 사용한 모델 학습 (문제 유형: binary)
# - 'URL' 컬럼은 column_types에서 텍스트 모달리티로 지정됩니다.
# - hyperparameters로 focal loss와 관련 설정을 적용합니다.
predictor = MultiModalPredictor(
    label='label',
    problem_type='binary',
    eval_metric='roc_auc',
    validation_metric='roc_auc'
)

predictor.fit(
    train_data=train_df,
    presets='best_quality',
    time_limit=None,  # 시간 제한 없음
    column_types={'URL': 'text'},
    seed=42,
    hyperparameters={
        "model.hf_text.checkpoint_name": "r3ddkahili/final-complete-malicious-url-model",
        "env.per_gpu_batch_size": 32,
        "optimization.patience": 3,
        "optimization.loss_function": "focal_loss",
        "optimization.focal_loss.alpha": weights,
    }
)

print("\nLeaderboard:")
print(predictor.leaderboard())

#%% md
# ## 5. Inference & Submission

#%%
# 테스트 데이터에 대해 예측 수행
test_pred_proba = predictor.predict_proba(test_df)

# 이진 분류의 경우, 보통 클래스 1 (악성 URL)의 확률을 사용
if 1 in test_pred_proba.columns:
    prediction_scores = test_pred_proba[1]
else:
    prediction_scores = test_pred_proba.iloc[:, 1]

test_df['probability'] = prediction_scores

# 제출 파일 생성 (ID와 probability 컬럼)
submission = test_df[['ID', 'probability']]
submission.to_csv('./submission/submission_multimodal.csv', index=False)
print("Submission file created.")
