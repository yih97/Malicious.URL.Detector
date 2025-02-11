import pandas as pd
import numpy as np
import warnings
from sklearn.utils.class_weight import compute_class_weight
from autogluon.multimodal import MultiModalPredictor

warnings.filterwarnings('ignore')

# 데이터 로드
final_train = pd.read_csv('../../data/preprocessed_data/final_train.csv')
final_test = pd.read_csv('../../data/preprocessed_data/final_test.csv')

# column types 지정
column_types = {
   'URL': 'text',
   'label': 'categorical',
   'digit_ratio': 'numerical',
   'special_char_count': 'numerical',
   'subdomain_count': 'numerical',
   'length': 'numerical'
}

# train_df의 'label' 컬럼을 기준으로 balanced 가중치 계산
weights = compute_class_weight(
   class_weight='balanced',
   classes=np.unique(final_train['label']),
   y=final_train['label'].values
)
# 가중치를 정규화 (합이 1이 되도록)
weights = weights / weights.sum()
weights = list(weights)
print("\nComputed class weights:", weights)

# MultiModalPredictor 생성 및 학습
predictor = MultiModalPredictor(
   label='label',
   problem_type='binary',
   eval_metric='roc_auc',
   validation_metric='roc_auc'
)

predictor.fit(
   train_data=final_train,
   column_types=column_types,
   presets='best_quality',
   time_limit=None,
   seed=42,
   hyperparameters={
      "model.hf_text.checkpoint_name": "r3ddkahili/final-complete-malicious-url-model",
      "env.per_gpu_batch_size": 32,
      "optimization.patience": 3,
      "optimization.loss_function": "focal_loss",
      "optimization.focal_loss.alpha": weights,
   }
)

# 특성 활용 확인
print("\n특성 메타데이터:")
print(predictor.feature_metadata_)

# 리더보드 확인
print("\n리더보드 (특성 중요도):")
leaderboard = predictor.leaderboard(silent=True)
print(leaderboard)

# 특성 중요도 시각화 (가능한 경우)
try:
   feature_importance = predictor.feature_importance()
   print("\n특성 중요도:")
   print(feature_importance)
except:
   print("\n특성 중요도를 계산할 수 없습니다.")

# 모델 예측
test_pred_proba = predictor.predict_proba(final_test)
if 1 in test_pred_proba.columns:
   prediction_scores = test_pred_proba[1]
else:
   prediction_scores = test_pred_proba.iloc[:, 1]

# 제출 파일 생성
submission = pd.DataFrame({
   'ID': final_test['ID'],
   'probability': prediction_scores
})
submission.to_csv('./submission/FE_multimodal.csv', index=False)
print("\nSubmission file created.")

