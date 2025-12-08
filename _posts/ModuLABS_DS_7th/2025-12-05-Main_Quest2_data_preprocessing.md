---
title: "[Main Quest] 비정제 데이터 체험하기"
description: "신용거래 이상탐지 데이터 다루기"
date: 2025-12-05 
categories: [ModuLABS DS 7th, Main Quest]
tags: [ModuLABS, Main Quest, SQL]
math: true
# pin: true
# 게시글 썸네일
# image :
# path : /commons/devices-mockup.png
---

# Project : 신용거래 이상탐지 데이터 다루기

## 1. 개요 및 데이터 불러오기

목표 : 신용카드 거래 데이터를 분석하여 이상 거래를 탐지
데이터 : `fraud.csv`
라이브러리 : `pandas`, `numpy`, `seaborn`, `geopy`

## 2. 데이터 기본 탐색 및 정리

데이터를 불러온 후 `.info()`, `.describe()` 등을 통해 데이터의 전반적인 구조와 통계적 특성을 파악

![MQ2-info](/assets/img/MQ2-info.png)


### 2.1 불필요한 컬럼 제거

식별자 역할을 하거나 분석에 직접적인 영향을 주지 않는 컬럼들을 삭제

- 삭제된 컬럼: `merchant`, `first`, `last`, `street`, `city`, `state`, `zip`, `job`, `trans_num`, `unix_time`


## 3. 피처 엔지니어링 (Feature Engineering)

데이터의 특성을 더 잘 반영하기 위해 기존 변수를 변환하거나 새로운 변수를 생성

### 3.1 Z-Score 기반 이상치 탐지 (Amount)

거래 금액(`amt`)이 평소 거래 패턴과 얼마나 다른지 파악하기 위해 Z-Score를 계산

- 각 카드 번호(`cc_num`)별 거래 금액의 평균과 표준편차를 계산
- 이를 바탕으로 전체 거래 금액(`amt`)에 대한 Z-Score(`amt_z`)를 생성
- 추가적으로 카드 번호와 거래 카테고리(`category`)를 조합한 그룹별 Z-Score(`cat_amt_z`)도 생성하여 더 세밀한 이상 탐지 지표 생성
- Z-Score 계산 후 불필요해진 mean, std 컬럼은 삭제

```python
# 카드별 거래 금액 통계 계산 및 병합
amt_info = cc_df.groupby('cc_num')['amt'].agg(['mean','std']).reset_index()
cc_df = cc_df.merge(amt_info, on='cc_num', how='left')

# amt Z-Score 계산
cc_df['amt_z'] = (cc_df['amt'] - cc_df['mean']) / cc_df['std']

# (카드+카테고리)별 거래 금액 통계 계산 및 병합
cat_info = cc_df.groupby(['cc_num', 'category'])['amt'].agg(['mean','std']).reset_index()
cc_df = cc_df.merge(cat_info, on=['cc_num', 'category'], how='left')

# 카테고리별 amt Z-Score 계산
cc_df['cat_amt_z'] = (cc_df['amt'] - cc_df['mean']) / cc_df['std']
```

### 3.2 시간 관련 피처 생성

- 거래 일시(`trans_date_trans_time`)에서 시간(`hour`) 정보를 추출하여 새로운 피처를 생성
- 거래 시간을 **'morning', 'afternoon', 'night', 'evening'** 4개 구간으로 범주화(`hour_cat`)
- 각 카드 번호별 시간대별 거래 비율(`hour_perc`)계산

```python
# 시간 추출 및 범주화
cc_df['hour'] = pd.to_datetime(cc_df['trans_date_trans_time']).dt.hour
cc_df['hour_cat'] = cc_df['hour'].apply(hour_func)

# 시간대별 거래 비율 계산 및 병합
# ... (중략) ...
cc_df = cc_df.merge(hour_cnt, on=['cc_num', 'hour_cat'], how='left')
```

### 3.3 거리(Distance) 피처 생성

- 사용자의 위치(`lat`, `long`)와 상점의 위치(`merch_lat`, `merch_long`) 정보를 이용하여 두 지점 사이의 거리를 계산
- `geopy` 라이브러리를 사용하여 거리를 계산하고 `distance` 컬럼에 저장
- 거래 금액과 마찬가지로, 각 카드 번호별 거리의 평균과 표준편차를 구해 거리 Z-Score(`dist_z`)를 생성

```python
from geopy.distance import distance

# 거리 계산
cc_df['distance'] = cc_df.apply(lambda x: distance((x['lat'], x['long']), (x['merch_lat'], x['merch_long'])).km, axis=1)

# 거리 Z-Score 계산
# ... (중략) ...
cc_df['dist_z'] = (cc_df['distance'] - cc_df['mean']) / cc_df['std']
```

### 3.4 나이(Age) 정보 변환

생년월일(`dob`) 컬럼을 날짜 형식으로 변환 후, 연도(`year`) 정보만 추출하여 수치형 데이터로 변환

```python
cc_df['dob'] = pd.to_datetime(cc_df['dob']).dt.year
```

### 3.5 범주형 데이터 인코딩 (One-Hot Encoding)

거래 카테고리(`category`)와 같은 범주형 데이터를 모델이 학습할 수 있도록 원-핫 인코딩(One-Hot Encoding)을 적용

```python
cc_df = pd.get_dummies(cc_df, columns=['category'], drop_first=True)
```