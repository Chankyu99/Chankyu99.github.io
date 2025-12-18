---
title: "[Project] Data Transformation"
description: "영국 시장의 중고 자동차 가격 데이터 다루기"
date: 2025-12-03
categories: [Project, Personal]
tags: [Project, DataAnalysis]
mermaid: true
# pin: true
# 게시글 썸네일
# image :
# path : /commons/devices-mockup.png
---

# Target : 적절한 데이터 전처리와 스케일링,PCA 기법 적용


# Code
```python
import pandas as pd         # Module Import
import numpy as np


car_df = pd.read_csv('cars.csv')        # 데이터 업로드
brand_df = pd.read_csv('brand.csv')


car_df['brand'] = car_df['title'].str.split().str[0]        # object를 숫자형으로 활용
brand_df['title'] = brand_df['title'].str.upper()
brand_df = brand_df.rename(columns={'title': 'brand'})

merged_df = car_df.merge(brand_df, on='brand', how='left')  # 데이터 결합

print(f"중복 제거 전: {len(merged_df)}행")
merged_df = merged_df.drop_duplicates()     # 모든 컬럼에 대해 중복된 행 제거
print(f"중복 제거 후: {len(merged_df)}행")

country_stats = merged_df.groupby('country')['Price'].agg(['mean', 'count']).reset_index() # 데이터 그룹화 및 요약

print("\n[국가별 평균 가격 및 매물 수]")    # 그룹화 결과 확인
pd.DataFrame(country_stats)
```
![output1](/assets/img/output1.png)

```python
from sklearn.preprocessing import RobustScaler  # RobustScaler import

# 수치형 데이터 전처리
merged_df['Engine'] = merged_df['Engine'].str.split('L').str[0]
merged_df['Engine'] = pd.to_numeric(merged_df['Engine'], errors='coerce')

# 분석에 사용할 수치형 컬럼 선정 및 결측치 제거
# merged_df.info()
numeric_features = ['Price', 'Mileage(miles)', 'Registration_Year', 'Previous Owners', 'Engine', 'Doors', 'Seats']
df_clean = merged_df.dropna(subset=numeric_features)

X = df_clean[numeric_features]

# RobustScaler 적용
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 스케일링된 데이터 확인
df_scaled = pd.DataFrame(X_scaled, columns=numeric_features)
print("\n[Robust Scaling 완료된 데이터 상위 5행]")
print(df_scaled.head())
```
![output2](/assets/img/output2.png)


```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA 적용
pca = PCA()
pca.fit(X_scaled)

# 설명된 분산 비율 (전체 데이터 분산 중 어느 비율정도까지 설명하는 가?)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)   # 누적 설명력 비율 -> 주성분 개수를 더할 때 마다 설명력이 증가함

print("\n[누적 설명력 비율]")
print(cumulative_variance)

# 적절한 주성분 개수 결정 (90% 기준)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"\n>> 전체 분산의 90% 이상을 설명하기 위한 주성분 개수: {n_components_90}개")

# Plot 시각화
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.9, color='r', linestyle='-', label='90% Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Scree Plot')
plt.legend()
plt.grid()
plt.show()

# 최종 PCA 변환
pca_final = PCA(n_components=n_components_90)
X_pca = pca_final.fit_transform(X_scaled)

pca_columns = [f'PC{i+1}' for i in range(n_components_90)]
df_pca_result = pd.DataFrame(X_pca, columns=pca_columns)
print(f"\n[최종 PCA 변환 데이터 (PC1 ~ PC{n_components_90})]")
print(df_pca_result.head())
```
![output3](/assets/img/output3.png)

# Review

## RobustScaling을 적용한 이유

이상치를 판단하는 과정에서 해당 데이터의 경우 StandardScaling을 적용하면 극단적인 이상치의 영향을 배제할 수 없어 스케일링 결과의 만족도가 낮았다. 그래서 StandardScaling의 이러한 단점을 보완할 수 있는 RobustScaling을 적용하였다.

## 중복 데이터 처리가 주성분 분석에 끼치는 영향

노드를 읽어가며 코드를 작성하는데 중복 데이터를 제거하는 코드가 없었다. 그래서 이미 중복 데이터가 걸러진 정제 데이터를 사용하는 줄 알았는데, 메서드를 활용해 확인해보니 중복데이터가 존재하였고, 이를 제거하는 코드를 추가했다. 

중복 데이터를 제거 전/후로 나누어 주성분 분석의 `explained_variance_ratio` 값을 비교해보았다. 
비교 결과, 약 -2% 내외로 차이가 발생하였다. 이는 사소한 차이지만, 더 많은 데이터를 사용할 경우 무시할 수 없는 영향력을 가진다. 
따라서 본 프로젝트에서는 중복값 제거를 추가로 진행하였다.