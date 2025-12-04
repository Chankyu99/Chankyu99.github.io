---
title: "피처 스케일링(Feature Scaling) & Encoding" 
description: "Min-Max, Z-Score부터 범주형 데이터 인코딩까지"
date: 2025-11-27
categories: [Development, Data Science]
tags: [Feature Scaling, Normalization, Standardization, Encoding]
math: true
---
**피처 스케일링**(Feature Scaling)은 머신러닝 모델을 학습하기 전에 서로 다른 범위의 수치형 데이터를 **같은 척도로 변환하는 전처리 기법**이다. 이번에는 [SQL 프로젝트](https://chankyu99.github.io/posts/Node_SQL_Project/)에서 사용했던 `StandardScaler`의 원리와 특징을 정리하고, 다양한 피처 스케일링 방법들을 정리해보았다. 그리고 피처 스케일링에 필요한 범주형 데이터와 연속형 데이터에 대해 알아보았다.

---

# 1. 피처 스케일링의 주요 종류 및 원리

## 1.1 선형 스케일링 (Min-Max Normalization)

**정의**: 데이터를 특정 범위(보통 0~1)로 압축하는 기법

$$ 
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

* $x$ : 원본 데이터 값
* $x_{min}$ : 데이터의 최소값
* $x_{max}$ : 데이터의 최대값

**특징**:

- **원본 분포 보존**: 데이터의 상대적 차이가 유지된다.
- **직관성**: 이해하고 구현하기 쉽다.
- **이상치에 민감**: 극단 값이 전체 스케일에 큰 영향을 미친다.
- **범위 안정성**: 상한과 하한이 명확하고 시간에 따라 변하지 않을 때 적합하다.

**사용 시기**:

- **k-NN**(k-Nearest Neighbors)과 같은 거리 기반 알고리즘
- 신경망
- 데이터가 균등 분포를 따를 때

---

## 1.2 표준화 (Z-score Standardization)

**정의**: 데이터를 평균 0, 표준편차 1로 변환하는 기법

$$
x' = \frac{x - \mu}{\sigma}
$$

* $x$ : 원본 데이터 값
* $\mu$ : 데이터의 평균
* $\sigma$ : 데이터의 표준편차

**특징**:

- **정규분포 가정**: 데이터가 정규분포(가우시안)를 따를 때 최적이다.
- **평균 중심화**: 변환된 데이터가 0 주위에 분포
- **일관된 이상치 처리**: Min-Max 스케일링보다 극단값에 대해 더 강건하다.
- **해석 가능성 상실**: 원래 단위가 사라진다.

**사용 시기**:

- SVM(Support Vector Machine)
- 로지스틱 회귀
- PCA(주성분 분석)
- 데이터가 정규분포를 따르거나 그에 가까울 때

---

## 1.3 로그 스케일링 (Log Scaling)

**정의**: 데이터에 자연로그를 적용하여 변환하는 기법

$$
x' = \ln(x)
$$

* $x$ : 원본 데이터 값

**특징**:

- **멱법칙 분포 처리**: 긴 꼬리 분포(skewed distribution)를 갖는 데이터에 효과적이다.
- **범위 압축**: 매우 넓은 범위의 데이터를 관리 가능한 범위로 압축한다.
- **양수만 지원**: 음수 또는 0 값에는 적용할 수 없다.

**사용 시기**:

- 영화 평점(많은 영화는 평점이 적고, 소수의 인기 영화는 많은 평점)
- 책 판매량(대부분의 책은 적게 팔리고 베스트셀러는 많이 팔림)
- 소셜 미디어 팔로워 수

***

## 1.4 클리핑 (Clipping)

**정의**: 극단 이상치의 값을 사전에 정의된 상한/하한 값으로 제한하는 기법이다.

**개념**:

- 특정 임계값(예: 3 표준편차)을 초과하는 값을 해당 임계값으로 설정한다.

**특징**:

- **이상치 강제 제한**: 극단적 아웃라이어의 영향을 제한한다.
- **정보 손실**: 극단 이상치가 실제로 중요할 수 있으므로 신중하게 사용해야 한다.

**사용 시기**:

- Z-score 스케일링 후 ±3을 초과하는 값을 제한할 때
- 극단 이상치가 있으면서 다른 스케일링과 함께 적용

***

## 1.5 로버스트 스케일링 (Robust Scaling)

**정의**: 중위수(median)와 사분위수 범위(IQR)를 이용하여 스케일링하는 기법이다.

**공식**:

$$
x' = \frac{x - \text{median}}{IQR}
$$

* $x$ : 원본 데이터 값
* $\text{median}$ : 데이터의 중위수
* $IQR$ : 데이터의 사분위수 범위

**특징**:

- **이상치 강건성**: 평균과 표준편차 대신 중위수와 IQR을 사용하여 극단 이상치의 영향을 최소화한다.
- **중간 50% 기준**: 데이터의 중간 50%의 범위를 기준으로 스케일링한다.
- **비정규 분포 대응**: 심하게 비대칭(skewed)인 데이터에 효과적이다.

**사용 시기**:

- 데이터에 많은 이상치가 포함되어 있을 때
- 데이터가 비정규 분포를 따를 때
- StandardScaler 대신 사용할 수 있습니다.

***

## 1.6 최대 절댓값 스케일링 (Maximum Absolute Scaling)

**정의**: 각 피처의 최댓값의 절댓값으로 나누어 [-1, 1] 범위로 스케일링한다.

**공식**:

$$
x' = \frac{x}{|x_{max}|}
$$

* $x$ : 원본 데이터 값
* $x_{max}$ : 데이터의 최대값

**특징**:

- **양수/음수 보존**: 부호를 유지하면서 크기만 정규화한다.
- **희소 데이터 친화적**: 0 값이 많은 희소 데이터에서 0을 유지한다.
- **이상치 민감**: 단일 극값이 전체 스케일에 큰 영향을 미친다.

**사용 시기**:

- 희소 데이터 처리
- 양수와 음수가 모두 있는 데이터
- 0-중심 데이터

***

## 1.7 벡터 단위 길이 스케일링 (Unit Vector Scaling / L2 Normalization)

**정의**: 각 샘플(행)을 단위 노름(norm = 1)으로 정규화하는 기법이다.

**공식**:

$$
x'_i = \frac{x_i}{\sqrt{\sum_{j} x_j^2}}
$$

* $x_i$ : i번째 샘플
* $x_j$ : j번째 피처

**특징**:

- **샘플 단위 처리**: 피처 단위가 아닌 샘플(관측치) 단위로 정규화한다.
- **방향 중심**: 크기가 아닌 방향이 중요할 때 사용한다.
- **텍스트 처리 표준**: 텍스트 마이닝과 추천 시스템에서 자주 사용된다.

**사용 시기**:

- 텍스트 분류
- 코사인 유사도 계산
- 추천 시스템
- 정보 검색

***

## 1.8 평균 정규화 (Mean Normalization)

**정의**: 데이터를 [-1, 1] 범위로 정규화하되, 평균을 기준으로 조정하는 기법이다.

**공식**:

$$
x' = \frac{x - \mu}{x_{max} - x_{min}}
$$

* $x$ : 원본 데이터 값
* $\mu$ : 데이터의 평균
* $x_{max}$ : 데이터의 최대값
* $x_{min}$ : 데이터의 최소값

**특징**:

- **평균 중심화**: 범위를 제한하면서 평균을 고려한다.
- **Min-Max와 Z-score의 중간 형태**

***

## 1.9 비선형 변환: Power Transformer

### 1.9.1 Box-Cox 변환

**정의**: 매개변수 λ를 최적화하여 데이터를 정규분포에 가깝게 변환하는 기법이다.

**특징**:

- **양수만 지원**: 0과 음수를 포함할 수 없습니다.
- **매개변수 기반**: 최대우도 추정을 통해 λ를 결정합니다.

**사용 시기**: 로그정규분포 데이터

***

### 1.9.2 Yeo-Johnson 변환

**정의**: Box-Cox와 유사하지만 **음수와 0을 포함한 데이터도 처리 가능**한 기법이다.

**특징**:

- **더 범용적**: 음수/양수/0 모두 처리
- **더 유연한 적용**

***

## 1.10 Quantile Transformer

**정의**: 데이터의 누적 분포 함수를 이용하여 모든 데이터를 균등 분포 또는 정규분포로 매핑한다.

**특징**:

- **비모수적(Non-parametric)**: 데이터의 실제 분포에 따라 유연하게 변환한다.
- **극단 이상치 강건**: 데이터 순위를 유지한다.
- **작은 데이터셋에서 과적합**: 수백 개 미만의 샘플에서는 과적합 위험이 있다.

**vs. Power Transformer**:

- Power Transformer는 **매개변수 기반**이고 해석 가능성이 더 높다.
- Quantile Transformer는 **비모수 기반**으로 더 유연하다다.

***

## 2. 연속형 데이터 (Continuous Data)

### 2.1 정의 및 특징

**연속형 데이터**는 실수 범위의 **무한히 많은 값**을 가질 수 있는 데이터이다.

**예시**:

- 키, 체중, 온도
- 연령, 소득
- 신호 진폭, 시간

### 2.2 연속형 데이터에 적용하는 스케일링

| 스케일링 기법 | 적합성 | 이유 |
|---|---|---|
| **선형 스케일링** | ⭐⭐⭐⭐ | 범위가 명확하고 극단 이상치가 적을 때 |
| **Z-score** | ⭐⭐⭐⭐⭐ | 정규분포를 따르는 대부분의 연속형 데이터 |
| **로버스트 스케일링** | ⭐⭐⭐⭐ | 이상치가 많을 때 |
| **로그 스케일링** | ⭐⭐⭐ | 멱법칙 분포(소득, 도시 인구) |
| **Power Transformer** | ⭐⭐⭐⭐ | 매우 비대칭한 데이터 |

***

## 3. 범주형 데이터 (Categorical Data)

### 3.1 정의 및 특징

**범주형 데이터**는 **미리 정의된 카테고리**에서만 값을 갖는 이산형 데이터이다.

**예시**:

- 성별 (남/여)
- 색상 (빨강/초록/파랑)
- 지역 (서울/부산/대구)
- 브라우저 (Chrome/Firefox/Safari)

### 3.2 범주형 데이터 처리 방법 (스케일링과 다른 인코딩)

**중요 구분**: 범주형 데이터는 스케일링하지 않고 **인코딩(Encoding)**한다.

#### 3.2.1 서수 인코딩 (Ordinal Encoding)

**정의**: 각 카테고리에 정수를 할당한다.

**예시**:

- "Low" → 0, "Medium" → 1, "High" → 2
- "male" → 0, "female" → 1

**사용 시기**: **순서가 있는 카테고리**(Ordinal Categories)

- 교육 수준: 초등학교 < 중학교 < 고등학교 < 대학교

**코드 예시**:

```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X = [['male'], ['female'], ['female']]
enc.fit_transform(X)
```

***

#### 3.2.2 원-핫 인코딩 (One-Hot Encoding)

**정의**: 각 카테고리를 별도의 이진 열로 변환한다.      

**예시**:

원본 데이터: ["Red", "Green", "Blue"]

변환 후:

| Red | Green | Blue |
|-----|-------|------|
| 1   | 0     | 0    |
| 0   | 1     | 0    |
| 0   | 0     | 1    |

**사용 시기**: **순서가 없는 카테고리**(Nominal Categories)

- 색상, 브라우저 종류, 지역

**특징**:

- 선형 모델(로지스틱 회귀, SVM)과 호환성이 우수하다.
- 고차원성(High Cardinality) 카테고리는 차원 폭증(Curse of Dimensionality) 위험이 있다.

**코드 예시**:

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X = [['male', 'from US'], ['female', 'from Europe']]
enc.fit_transform(X).toarray()
```

***

#### 3.2.3 타겟 인코딩 (Target Encoding)

**정의**: 각 카테고리를 해당 카테고리의 **목표 변수 평균값**으로 인코딩한다.

**특징**:

- 고차원 카테고리(zip code, 지역명)에 특히 효과적이다.
- 정보 누수(Data Leakage) 위험을 방지하기 위해 교차 검증(Cross-Validation)을 사용한다.

***

#### 3.2.4 라벨 인코딩 (Label Encoding)

**정의**: 각 고유 카테고리에 임의의 정수를 순차 할당한다.

**특징**:

- 서수 인코딩과 유사하지만 순서 의미 없음
- 나무 기반 모델(Random Forest, XGBoost)과 호환성 우수

***

## 4. 연속형 vs. 범주형 데이터 처리 비교

| 특성 | 연속형 데이터 | 범주형 데이터 |
|---|---|---|
| **값 유형** | 실수 범위 내 무한 값 | 유한한 사전 정의 카테고리 |
| **처리 목표** | 수치 척도 통일 (Scaling) | 숫자로 변환 (Encoding) |
| **대표 기법** | Z-score, Min-Max | One-Hot, Ordinal |
| **모델 적합성** | 모든 모델 | 트리 기반 모델, 선형 모델 |
| **메모리 효율** | 기본 메모리 사용 | One-Hot 적용 시 고차원화 가능 |
| **시계열 적용** | O | 제한적 |

***

## 5. 실무 선택 가이드

| 상황 | 추천 기법 |
|---|---|
| 정규분포 연속 데이터 | **Z-score Standardization** |
| 균등 분포 데이터 | **Linear Scaling (Min-Max)** |
| 많은 이상치 포함 | **Robust Scaling** |
| 멱법칙 분포 (매우 비대칭) | **Log Scaling** or **Power Transformer** |
| 거리 기반 알고리즘(k-NN) | **Linear Scaling** |
| 그래디언트 기반 알고리즘(SVM, LR) | **Z-score** |
| 희소 데이터 | **Max Absolute Scaling** |
| 텍스트/추천 시스템 | **Unit Vector Scaling** |
| 순서 있는 범주형 | **Ordinal Encoding** |
| 순서 없는 범주형 | **One-Hot Encoding** |
| 고차원 범주형(많은 카테고리) | **Target Encoding** |

***

## 결론

피처 스케일링은 **데이터의 특성과 사용 모델에 따라 신중하게 선택**해야 한다. **연속형 데이터는 스케일링**, **범주형 데이터는 인코딩**하되, 데이터의 분포와 이상치 존재 여부를 사전에 시각화하고 검토하는 것이 중요하다.

---
# 참고자료

[1](https://www.datacamp.com/tutorial/normalization-vs-standardization)
[2](https://developers.google.com/machine-learning/crash-course/numerical-data/normalization)
[3](https://ethans.co.in/blogs/different-types-of-feature-scaling-and-its-usage/)
[4](https://scikit-learn.org/stable/modules/preprocessing.html)
[5](https://towardsdatascience.com/why-is-feature-scaling-important-in-machine-learning-discussing-6-feature-scaling-techniques-2773bda5be30/)
[6](https://www.artech-digital.com/blog/ultimate-guide-to-feature-scaling-in-ml)
[7](https://www.geeksforgeeks.org/machine-learning/feature-engineering-scaling-normalization-and-standardization/)
[8](https://atalupadhyay.wordpress.com/2025/02/10/normalization-and-standardization-scaling-your-way-to-machine-learning-success/)
[9](https://mkang32.github.io/python/2020/12/27/feature-scaling.html)
[10](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html)
[11](https://stackoverflow.com/questions/73335665/differences-between-quantiletransformer-and-powertransformer)
[12](https://www.sciencedirect.com/science/article/pii/S2666285X22000565)
[13](https://www.blog.trainindata.com/mastering-data-preprocessing-techniques/)
[14](https://stackoverflow.com/questions/43554821/feature-preprocessing-of-both-continuous-and-categorical-variables-of-integer-t)
[15](https://metadesignsolutions.com/feature-engineering-in-machine-learning/)
[16](https://www.almabetter.com/bytes/articles/data-processing)
[17](https://www.youtube.com/watch?v=lV_Z4HbNAx0)