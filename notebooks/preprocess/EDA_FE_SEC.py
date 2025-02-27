# %%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action='ignore')

# %%
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# %%
# '[.]'을 '.'으로 복구
train['URL'] = train['URL'].str.replace(r'\[\.\]', '.', regex=True)
test['URL'] = test['URL'].str.replace(r'\[\.\]', '.', regex=True)

# %%
# URL 구조 세분화 전처리 과정 추가
from urllib.parse import urlparse
import tldextract


def extract_url_features(df):
    # URL 기본 구성요소 추출: scheme, netloc, path, params, query, fragment
    parsed_urls = df['URL'].apply(urlparse)
    df['scheme'] = parsed_urls.apply(lambda x: x.scheme)
    df['netloc'] = parsed_urls.apply(lambda x: x.netloc)
    df['path'] = parsed_urls.apply(lambda x: x.path)
    df['params'] = parsed_urls.apply(lambda x: x.params)
    df['query'] = parsed_urls.apply(lambda x: x.query)
    df['fragment'] = parsed_urls.apply(lambda x: x.fragment)

    # tldextract를 이용하여 도메인 세분화: 서브도메인, 도메인, TLD 추출
    extracted = df['URL'].apply(tldextract.extract)
    df['subdomain_text'] = extracted.apply(lambda x: x.subdomain)
    df['domain_text'] = extracted.apply(lambda x: x.domain)
    df['suffix_text'] = extracted.apply(lambda x: x.suffix)

    # 추가 피처: 경로 내 세그먼트 개수, 쿼리 파라미터 개수
    df['path_segment_count'] = df['path'].apply(lambda x: len([seg for seg in x.split('/') if seg]))
    df['query_param_count'] = df['query'].apply(lambda x: len(x.split('&')) if x else 0)

    return df


# train, test 데이터에 URL 구조 세분화 적용
train = extract_url_features(train)
test = extract_url_features(test)

# %%
## 새로운 변수 생성
# URL 길이
train['length'] = train['URL'].str.len()
test['length'] = test['URL'].str.len()

# 서브도메인 개수 (기존 방식)
train['subdomain_count'] = train['URL'].str.split('.').apply(lambda x: len(x) - 2)
test['subdomain_count'] = test['URL'].str.split('.').apply(lambda x: len(x) - 2)

# 특수 문자('-', '_', '/') 개수
train['special_char_count'] = train['URL'].apply(lambda x: sum(1 for c in x if c in '-_/'))
test['special_char_count'] = test['URL'].apply(lambda x: sum(1 for c in x if c in '-_/'))

# 디지털 문자 관련
train['digit_count'] = train['URL'].str.count(r'\d')
test['digit_count'] = test['URL'].str.count(r'\d')
train['digit_ratio'] = train['digit_count'] / train['length']
test['digit_ratio'] = test['digit_count'] / test['length']

# 대문자 관련
train['uppercase_count'] = train['URL'].str.count(r'[A-Z]')
test['uppercase_count'] = test['URL'].str.count(r'[A-Z]')
train['uppercase_ratio'] = train['uppercase_count'] / train['length']
test['uppercase_ratio'] = test['uppercase_count'] / test['length']

# 추가 특수문자
train['abnormal_chars'] = train['URL'].str.count(r'[^a-zA-Z0-9\-\./_]')
test['abnormal_chars'] = test['URL'].str.count(r'[^a-zA-Z0-9\-\./_]')
train['dots_count'] = train['URL'].str.count(r'\.')
test['dots_count'] = test['URL'].str.count(r'\.')

# URL 구조 관련 기존 피처
train['path_length'] = train['URL'].apply(lambda x: len(x.split('/')[-1]) if '/' in x else 0)
test['path_length'] = test['URL'].apply(lambda x: len(x.split('/')[-1]) if '/' in x else 0)

train['query_count'] = train['URL'].str.count(r'\?')
test['query_count'] = test['URL'].str.count(r'\?')

train['and_count'] = train['URL'].str.count(r'\&')
test['and_count'] = test['URL'].str.count(r'\&')

# %%
# 상관계수 계산
feature_cols = ['length', 'subdomain_count', 'special_char_count',
                'digit_count', 'digit_ratio', 'uppercase_count', 'uppercase_ratio',
                'abnormal_chars', 'dots_count', 'path_length', 'query_count', 'and_count']

correlation_matrix = train[feature_cols + ['label']].corr()

# label과의 상관관계 확인
label_corr = correlation_matrix['label'].abs().sort_values(ascending=False)
print("\n특성과 label의 상관관계 (절대값 기준 내림차순):")
print(label_corr)

# 상관계수 0.3 이상인 특성 선택
selected_features = label_corr[label_corr >= 0.3].index.tolist()
selected_features.remove('label')  # label 제외
print("\n선택된 특성:", selected_features)

# 최종 데이터셋 생성
final_train = train[['ID', 'URL', 'label'] + selected_features]
final_test = test[['ID', 'URL'] + selected_features]

print("\n최종 학습 데이터 shape:", final_train.shape)
print("최종 테스트 데이터 shape:", final_test.shape)
# %%
final_train.to_csv('../data/preprocessed_data/final_train.csv', index=False)
final_test.to_csv('../data/preprocessed_data/final_test.csv', index=False)
