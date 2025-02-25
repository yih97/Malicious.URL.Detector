# 악성 URL 분류 AI 경진대회

## 📌 개요
**월간 데이콘 '악성 URL 분류 AI 경진대회'**에 오신 것을 환영합니다! 이 대회에서는 URL 데이터의 텍스트 정보를 활용하여 악성 URL을 탐지하는 **AI 알고리즘을 개발**합니다.

인터넷이 발전함에 따라 악성 URL을 통한 사이버 공격이 증가하고 있으며, 이는 개인정보 유출, 금융 사기, 악성 코드 배포 등 다양한 피해를 초래합니다. 따라서 본 대회를 통해 **정확한 악성 URL 분류 모델을 개발하는 것**이 목표입니다.

---

## 🏆 평가 기준
### **리더보드 (Leaderboard)**
- **평가 지표:** ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- **Public Score:** 전체 테스트 데이터 중 사전 샘플링된 30%
- **Private Score:** 전체 테스트 데이터 중 나머지 70%

### **최종 평가 방법**
1. **1차 평가**: 리더보드 Private Score를 기준으로 선정
2. **2차 평가**: Private Score 상위 10팀이 코드 및 PPT 제출 → 코드 검증 후 최종 수상 결정

---

## 📝 대회 규칙
### **1. 참가 방법**
- 개인 또는 팀(최대 5명)으로 참여 가능
- 개인 참가: 팀 신청 없이 자유롭게 제출 가능
- 팀 참가: 팀 페이지에서 구성 가능 (단, 중복 팀 참가 불가)

### **2. 제출 제한**
- 1일 최대 제출 횟수: **5회**

### **3. 외부 데이터 및 사전 학습 모델**
- **외부 데이터 사용 금지** (경진대회 제공 데이터만 사용 가능)
- **공식 공개 사전 학습 모델** 사용 가능 (법적 제약 없는 경우)

### **4. 코드 및 PPT 제출 규칙**
#### **코드 제출**
- `/data` 경로에서 데이터 입출력 처리 필수
- 파일 확장자: `.R`, `.rmd`, `.py`, `.ipynb`
- 코드 및 주석은 **UTF-8 인코딩**
- 라이브러리 로딩 포함하여 **모든 코드가 오류 없이 실행 가능해야 함**
- **개발 환경(OS) 및 라이브러리 버전 명시 필수**
- **사전 학습 모델 사용 시 출처 및 다운로드 링크 기재**
- 제출한 코드로 **Private Score 재현 가능해야 함**

#### **PPT 제출**
- 자유 형식 (솔루션 설명 포함)
- 코드와 PPT 모두 제출해야 수상이 가능함

---

## 📂 데이터 소개
데이터는 URL 정보를 포함하며, **주어진 URL이 악성인지 여부를 분류하는 것이 목표**입니다.

[데이터 다운로드 링크](https://dacon.io/competitions/official/236451/data)

**데이터 예시:**
| URL | label |
|---|---|
| http://example.com/login | 0 (정상) |
| http://phishing-site.com | 1 (악성) |

**변수 설명:**
- `URL`: 분석할 대상 URL
- `label`: 악성 여부 (0 = 정상, 1 = 악성)

---

## 🏗 모델 개발 가이드라인
1. **데이터 전처리**
   - URL에서 의미 있는 특징(도메인, 길이, 특수문자 포함 여부 등) 추출
   - 토큰화, TF-IDF 또는 Word2Vec을 활용한 변환

2. **모델 학습**
   - 머신러닝 모델 (XGBoost, LightGBM, CatBoost 등)
   - 심층 신경망 (LSTM, CNN 기반 모델) 사용 가능

3. **모델 평가 및 튜닝**
   - ROC-AUC 점수를 기준으로 성능 최적화
   - 하이퍼파라미터 튜닝 (Optuna, Grid Search 등 활용)

4. **결과 분석 및 시각화**
   - Feature Importance 분석
   - 혼동 행렬, Precision-Recall 곡선 등 활용

---

## ⚠️ 유의 사항
- **1일 최대 제출 횟수:** 5회
- **사용 가능 언어:** Python
- **외부 데이터 사용 금지** (제공된 데이터만 활용 가능)
- **팀 외의 코드 공유 금지 (데이콘 내 공개 공유만 가능)**
- **대회 종료 후 Private Score 검증을 통해 최종 순위 결정**

---

## 📞 문의 방법
- 대회 운영 및 데이터 관련 질문 가능
- **토크 페이지**에 공식 문의 글 작성 가능

---

## 🚀 목표
이번 대회에서는 **URL의 텍스트 정보만을 활용하여 악성 여부를 예측하는 AI 모델을 개발**합니다.
여러분의 창의적인 접근법으로 **인터넷 보안 강화에 기여할 수 있는 모델을 만들어 주세요!** 🔥
