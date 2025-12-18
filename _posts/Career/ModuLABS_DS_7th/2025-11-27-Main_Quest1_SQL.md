---
title: "[Main Quest] SQL을 사용하여 데이터베이스 다루기"
description: "고객 세그먼테이션"
date: 2025-11-27 
categories: [Career, ModuLABS DS 7th]
tags: [ModuLABS, Main Quest, SQL]
math: true
# pin: true
# 게시글 썸네일
# image :
# path : /commons/devices-mockup.png
---
이전 학습을 통해 우리는 고객의 구매 데이터를 기반으로 **RFM(Recency, Frequency, Monetary)** 테이블인 `user_data`를 생성했다. 이번 포스팅에서는 이 데이터를 활용해 **K-Means 클러스터링**을 수행하고, 실제 비즈니스에 적용 가능한 **고객 세그먼테이션(Segmentation) 전략**까지 도출해보려 한다.

단순한 코드 실행을 넘어, **데이터 전처리 → 차원 축소 → 모델링 → 해석**으로 이어지는 전체적인 분석 흐름과 그 안에서 얻을 수 있는 인사이트에 집중하자.

# 1. 이상 데이터 분석 및 처리

> ### ***이상치 데이터란?***
1986년 미국 전체 대학 졸업생의 평균 초봉이 **22,000달러**이다. 그런데 그중 한 대학의 학과별 졸업생 평균 초봉을 비교해보니 지리학과 졸업생의 초봉이 무려 전체 평균의 12배에 가까운 **250,000달러**로 가장 높았다. 이는 해당 학교의 지리학과 졸업생 중 농구의 전설, **마이클 조던**이 있었는데, 초봉 평균 계산에 그의 연 **70,000달러**짜리 NBA계약을 포함시켰기 때문이다. 여기서 마이클 조던의 초봉이 이상치 데이터라고 할 수 있다.
<br>
_조성준, "빅데이터 커리어 가이드북", p124_

## 이상치 데이터 vs 더티 데이터 구분하기

데이터 정제 과정을 거치지 않아 각종 오류를 포함하고 있는 데이터로, 아직 깔끔하게 정리되지 않아 지저분한 상태의 데이터라고 볼 수 있다. 더티 데이터에는 다양한 유형이 있다. 

### ***'최종 학력'에 대한 데이터를 수집 할 때 발생할 수 있는 각종 더티 데이터의 유형*** 

| 유형 | 예시 |
| :--- | :--- |
| **누락된 데이터**<br>(Missing Data) | 질문에 대답하지 않음 |
| **잘못된 데이터**<br>(Wrong Data) | '박사' 대신 '박박사' 기입 (오타) |
| **구식 데이터**<br>(Outdated Data) | 석사 과정을 마쳐 최종 학력이 '석사'지만, 업데이트하지 않아 여전히 '학사'로 기입 |
| **비표준 데이터**<br>(Non-Standard) | '박사'라는 통일된 표현 대신 'Doctoral Degree', 'Ph.D' 등 다른 표현을 혼용함 |
| **모호한 데이터**<br>(Ambiguous Data) | '대학교에서 교육받음' 등 여러 가지로 해석될 여지가 있는 정보 |

> **GIGO**(**G**arbage **I**n, **G**arbage **O**ut)라는 말이 있다. <br>쓰레기가 들어가면 쓰레기가 나온다는 의미로, 문장에서 데이터 전처리 과정이 얼마나 중요한가를 알 수 있다.


## 왜 하필 Z-Score일까? (장단점 정리)

이번 프로젝트에서 이상치를 탐지하기 위해 **Z-Score(표준화)** 기법을 선택했다.
데이터 전처리에서 가장 흔하게 쓰이는 방법이지만, 무턱대고 쓰기보다는 **왜 이걸 쓰는지, 쓸 때 주의할 점은 무엇인지** 알고 넘어가는 것이 중요하다.

### 1. Z-Score란?
데이터를 **평균($\mu$)이 0, 표준편차($\sigma$)가 1**인 분포로 변환하는 것이다.
쉽게 말해, **"이 데이터가 평균에서 얼마나 멀리 떨어져 있는가?"**를 수치화한 것이다.

$$Z = \frac{x - \mu}{\sigma}$$

* $x$: 원본 데이터 값
* $\mu$: 데이터의 평균
* $\sigma$: 데이터의 표준편차

### 2. Z-Score의 장단점 비교

| 구분 | 내용 |
| :--- | :--- |
| **장점** | • **서로 다른 단위 통합:** '구매 금액(원)'과 '방문 횟수(회)'처럼 단위가 다른 데이터를 동일한 척도로 비교할 수 있다.<br>• **이상치 탐지 용이:** 일반적으로 Z-Score가 **±3 이상**이면 통계적으로 매우 드문 값이므로 이상치로 판단하기 쉽다.<br>• **모델 성능 향상:** K-Means나 PCA처럼 '거리' 기반 알고리즘의 성능을 높여준다. |
| **단점** | • **정규분포 가정:** 데이터가 정규분포(종 모양)를 따를 때 가장 정확하다. 한쪽으로 치우친 데이터에서는 신뢰도가 떨어진다.<br>• **극단적 이상치에 민감:** 정말 말도 안 되는 거대 이상치가 있으면, 평균과 표준편차 자체가 왜곡되어 Z-Score 계산이 망가질 수 있다. |

> **결론:**
> 이번 데이터셋은 특성 간 단위 차이가 크기 때문에 **표준화(Standardization)**가 필수적이며, 분포를 확인했을 때 적용 가능한 수준이라 판단하여 Z-Score를 활용했다.

<br>
Z-Score는 일반적으로 평균이 0이고 표준 편차가 1인 표준 정규 분포를 기준으로 한다. 따라서 Z-Score가 0보다 크면 해당 데이터 포인트는 해당 특성의 평균보다 큰 값을 가지며, Z-Score가 0보다 작으면 해당 데이터 포인트는 해당 특성의 평균보다 작은 값을 가진다는 의미이다.

이상치 감지를 위해 Z-Score를 사용 시 일반적으로 특정 임계값(Threshold)를 설정해 이 임계값을 초과하는 Z-score를 가지는 데이터를 이상치로 간주한다. 임계값을 조정해 이상치 데이터의 범주를 설정할 수 있다.

우선 이상치를 찾기 위해 `user_data` 테이블을 불러오고 Z-score를 계산해보자.

```python
# 라이브러리 불러오기 
import pandas as pd 

# 데이터 불러오기
user_data = pd.read_csv('aiffel/customer_segmentation/user_data.csv')

# 필요한 라이브러리 불러오기
from scipy import stats
import numpy as np

# CustomerID를 제외한 나머지 특성(컬럼)을 정규화해 Z-score 계산
# Z-score 계산 
z_scores = stats.zscore(user_data.iloc[:, 1:], axis=0)  

# Z-score 절대값 계산
z_scores = np.abs(z_scores)

# Z-score 출력
z_scores

# 임계값 설정
threshold = 3 # Z-score가 3보다 큰 값은 이상치로 간주

# z-score 기준으로 이상치를 찾아서 outlier 컬럼에 이상치 여부 기입 (0: 정상, 1:이상치)
user_data['outlier'] = (z_scores > threshold).any(axis=1).astype(int)

# 시각화에 필요한 라이브러리 불러오기
import matplotlib.pyplot as plt 

# user_data['outlier']을 활용하여 이상치 여부에 따른 확률 계산
# value_counts()는 열의 고윳값의 개수를 반환하지만 normalize=True를 사용하면 열에 있는 값의 개수 비율(상대적 빈도)을 반환함
outlier_percentage = pd.value_counts(user_data['outlier'], normalize=True) * 100

# 시각화 자료 크기 조정
plt.figure(figsize=(3, 4))

# outlier_percentage라는 데이터로 bar chart 시각화
# x축 값을 0과 1로 지정
bars = plt.bar(['0', '1'], outlier_percentage)

# 퍼센트(%) 표시
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval/2, f'{yval:.2f}%', fontsize=10, va='center', ha='center')

plt.title('Normal(0) vs Outlier(1)') # 표 제목
plt.yticks(ticks=np.arange(0, 101, 10)) # y축 표기 (0~100까지 10단위로 증가)
plt.ylabel('Percentage (%)') # y축 범례
plt.xlabel('Outlier') # x축 범례
plt.show() # 출력
```

![시각화 결과 그래프 이미지](/assets/img/outlier.jpg)
_93.51%의 데이터가 정상치(0)으로 라벨링 되었다._

## 시각화 결과 정리
약 6%의 고객이 데이터셋에서 이상치로 식별되었다. 이 비율은 데이터의 중요한 부분을 잃지 않고, 잠재적으로 노이즈가 있는 데이터 포인트를 유지하기에 합리적인 비율로 보인다.

그래서 프로젝트 진행 시 이상치를 처리해 군집의 품질에 미치는 영향을 방지하는 것이 중요하다.

이제 이상치를 제거하고, 정제된 데이터를 만들면 군집화를 진행할 수 있다.

```python
# 이상치 제거
user_data = user_data[user_data['outlier'] == 0]

# Outlier 컬럼 삭제
user_data = user_data.drop(columns=['outlier'])

# DataFrame의 인덱스를 리셋하고, 이전 인덱스를 컬럼으로 추가X
user_data.reset_index(drop=True, inplace=True)

# 정제된 데이터프레임 확인
user_data.head()
```

# 변수 간 상관관계 분석

## 상관관계
상관관계 : 쉽게 말해 두 변수가 있을 때, 한 변수가 변할 때 다른 변수가 어떻게 변하는가를 보여주는 지표를 의미한다.

보통 상관관계는 -1과 1 사이의 값을 가지는데, 절댓값의 수치가 클 수록 강한 관계를 가진다.

- 양의 상관관계 : 한 변수가 증가할 때 다른 변수도 증가
- 음의 상관관계 : 한 변수가 증가할 때 다른 변수가 감소
- 상관관계 없음 : 한 변수가 증가하거나 감소해도 다른 변수는 관련이 없음
> 상관관계가 인과관계를 의미하는 것은 아님!! 

## 다중공선성
다중공선성 : 두 개 이상의 독립변수가 서로 높은 상관관계를 가지고 있을 때 발생하는 문제

이로 인해 각 변수의 영향력을 분리해서 추정하기 어려워지고, 작은 데이터의 변화에도 모델의 변수 예측값이 크게 달라질 수 있다. 또한 실제로 유의미한 결과를 도출하거나 해석하기 매우 어렵다.

그래서 이를 해결하기위해 다중공선성 문제가 있는 변수를 찾아 하나만을 선택하거나, 두 변수를 결합하여 새로운 변수를 만드는 방법으로 해결할 수 있다.
또는 후술할 차원 축소 기술을 사용해 새로운 변수를 생성한다.

상관계수를 시각화해 다중공선성 문제가 있는지 파악해보자. 

```python
# 시각화 라이브러리 불러오기
import seaborn as sns  

# 'CustomerID' 열을 제외(drop)하고 상관 관계 행렬 계산(corr())
corr = user_data.drop(columns=['CustomerId']).corr()

# 행렬이 대각선을 기준으로 대칭이기 때문에 하단만 표시하기 위한 마스크 생성
mask = np.zeros_like(corr) # np.zeros_like()는 0으로 가득찬 array 생성, 크기는 corr와 동일   
mask[np.triu_indices_from(mask, k=1)] = True # array의 대각선 영역과 그 윗 부분에 True가 들어가도록 설정

# 히트맵 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(corr, mask=mask, cmap='Greys', annot=True, fmt='.2f')
plt.show()
```

![heatmap](assets/img/heatmap.png)

나의 경우 가장 높은 상관관계를 가지는 것은 `total_transactions`와 `unique_products`가 0.95의 상관계수를 가지며, 그 뒤로 `user_total`과 `item_cnt`가 0.90의 상관계수를 가지는 것을 알 수 있었다.

이렇게 찾아낸 높은 상관계수를 가진 변수들을 이용해 중복되어 제거할 수 있는 특성이 있는지, 또는 다중공선성 문제가 있는 컬럼 쌍이 무엇인지를 알 수 있다.

# 피처(Feature) 스케일링
이상치를 제거하고, 변수 간 상관관계를 분석해 다중공선성 문제를 해결한 후 본격적인 클러스터링 및 차원 축소를 진행하기 전, 특성 값을 스케일링하는 것이 매우 중요하다.

## 피처 스케일링을 하는 이유
1. K-Means 클러스터링
데이터 포인트 간의 '거리' 개념에 크게 의존하여 군집을 형성하는 알고리즘인 K-Means의 경우 각 특성이 유사한 척도를 사용하지 않는다면, 값이 큰 특성이 클러스터링 결과에 불균형을 일으킬 수 있다. 따라서 특성을 스케일링하여 각 특성이 유사한 척도를 가지도록 하는 것이 중요하다.

2. 차원 축소 (PCA : 주성분 분석)
차원 축소 기술인 PCA는 주성분 분석을 통해 데이터의 주요 특성을 추출하여 차원을 축소하는 기술이다. 이 과정에서 각 특성이 유사한 척도를 가지도록 하면 가장 효율적으로 데이터를 압축할 수 있다.

## 피처 스케일링하기

> 목표 : 각 특성의 평균이 0이고 표준 편차가 1이되도록 변환

범주형 데이터가 아닌 연속형 데이터만 진행한다. 
현재 데이터에서 `CustomerId`가 유일한 범주형 데이터인데, 고객을 식별하기 위한 식별자라 클러스터링에 대한 의미 있는 정보를 포함하고 있지 않으므로 제외한다.

```python
# Standard Scaler 불러오기 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 원본 데이터에 영향을 주지 않기 위해 복사 
data = user_data.copy()

# CustomerID를 제외한 데이터에 스케일링 적용
columns_list = data.iloc[:, 1:].columns # iloc: 데이터 특정 값 추출, columns: 데이터프레임의 열 이름 조회 
data[columns_list] = scaler.fit_transform(data[columns_list])

# 스케일링된 데이터 확인
data.head()
```

![scale](assets/img/scaling.png)

> 
스케일링 결과, 피처들이 모두 양수였던 것들이 더 작아지고 음수인 값들이 생겼는데, <br>이는 StandardScaler가 각 컬럼 값들의 분포가 평균 0, 표준편차가 1이 되도록 스케일링하기 때문이다.

# 차원 축소 (Dimensionality Reduction)

차원 축소는 고차원 데이터의 특징을 보존하며 차원의 수를 줄이는 과정이다.

## 차원 축소를 하는 이유?

1. 다중공선성 식별
2. K-Means 클러스터링을 통한 더 나은 클러스터링
3. 노이즈 감소
4. 시각화 향상
5. 계산 효율성 향상

## 차원 축소 종류
1. PCA (Principal Component Analysis) 
2. ICA (Independent Component Analysis) 
3. ISOMAP (Isometric Mapping)
4. t-SNE (t-Distributed Stochastic Neighbor Embedding)
5. UMAP (Uniform Manifold Approximation and Projection)

여러가지 차원 축소 기법 중 프로젝트에 알맞은 차원 축소 기법을 선택하는 것이 중요하다.

## PCA (Principal Component Analysis)

이번 프로젝트에서는 데이터의 차원을 줄이기 위해 **PCA(Principal Component Analysis)** 기법을 사용한다. 

### 주성분(Principal Component)이란?
쉽게 말해 **"원본 데이터의 정보를 최대한 훼손하지 않으면서 압축한 새로운 변수"**다.
여기서 '정보'란 통계학적으로 **'분산(Variance)'**을 의미한다.

* **PC1 (첫 번째 주성분):** 데이터의 변동성(분산)을 가장 많이 설명하는 축. 즉, 데이터가 가장 넓게 퍼져 있는 방향이다.
* **PC2 (두 번째 주성분):** PC1이 설명하지 못한 나머지 변동성 중 가장 큰 부분을 설명하는 축. (PC1과 수직을 이룬다.)

### 2. PCA를 하는 이유 
수십, 수백 개의 특성(Feature)을 다 쓰면 좋겠지만, 현실적으로 계산 비용이 너무 크고 시각화도 불가능하다.

* **정보 압축:** 100개의 변수가 가진 정보의 95%를 단 5~6개의 주성분으로 요약할 수 있다면 훨씬 효율적이다.
* **노이즈 제거:** 덜 중요한 주성분(분산이 작은 축)을 버림으로써 데이터의 노이즈를 제거하는 효과가 있다.
* **시각화:** 3차원 이상의 고차원 데이터는 눈으로 볼 수 없지만, 2~3개의 주성분으로 줄이면 시각화하여 데이터의 패턴(군집)을 파악할 수 있다.

> **요약:**
> PCA는 고차원 데이터의 **'액기스(주요 특징)'**만 뽑아내어 저차원으로 축소하는 기술이다. 이를 통해 우리는 더 빠르고 효율적으로 분석을 수행할 수 있다.

## PCA 적용하기
```python
# PCA 불러오기  
from sklearn.decomposition import PCA

# CustomerID를 인덱스로 지정  
data.set_index('CustomerId', inplace=True)

# PCA 적용
pca = PCA().fit(data)

# Explained Variance의 누적합 계산 
# Explained Variance: 주성분이 데이터의 변동성을 얼마나 포착하는가를 나타내는 비율

explained_variance_ratio = pca.explained_variance_ratio_ # explained_variance_ratio_: Explained Variance 비율을 계산해 주는 함수
cumulative_explained_variance = np.cumsum(explained_variance_ratio) # cumsum: 각 원소의 누적합을 계산하는 함수

plt.figure(figsize=(15, 8)) 

# 각 성분의 설명된 분포에 대한 막대 그래프
barplot = sns.barplot(x=list(range(1, len(cumulative_explained_variance) + 1)), y=explained_variance_ratio, alpha=0.8)

# 누적 분포에 대한 선 그래프
lineplot, = plt.plot(range(0, len(cumulative_explained_variance)), cumulative_explained_variance, marker='o', linestyle='--', linewidth=2)

# 레이블과 제목 설정
plt.xlabel('Number of Components', fontsize=14)
plt.ylabel('Explained Variance', fontsize=14)
plt.title('Cumulative Variance vs. Number of Components', fontsize=18)

# 눈금 및 범례 사용자 정의
plt.xticks(range(0, len(cumulative_explained_variance)))
plt.legend(handles=[barplot.patches[0], lineplot],
           labels=['Explained Variance', 'Cumulative Explained Variance'])  

# 두 그래프의 분산 값 표시
x_offset = -0.3
y_offset = 0.01
for i, (ev_ratio, cum_ev_ratio) in enumerate(zip(explained_variance_ratio, cumulative_explained_variance)):
    plt.text(i, ev_ratio, f"{ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)
    if i > 0:
        plt.text(i + x_offset, cum_ev_ratio + y_offset, f"{cum_ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)

plt.grid(axis='both')   
plt.show()
```

![pca](assets/img/pca.png)

결과를 통해 각 주성분이 데이터 집합의 총 분산 중 얼마나 많은 부분을 설명하는지와 처음 n개의 주성분에 의해 설명되는 누적 분산을 보여준다.

- 첫 번째 주성분은 전체 분산의 약 47%를 설명한다.
- 첫 두 주성분은 약 64%의 분산을 함께 설명한다.
- 첫 세 개의 주성분은 전체 분산의 약 75%를 설명한다.

최적의 주성분 수를 선택하기 위해 일반적으로 ***Cumulative Explained Variance(누적 설명 분산)***이 크게 증가하지 않는 지점을 찾는다. 이를 곡선의 **'엘보우 포인트'**라고 한다. 그림에서 볼 수 있듯이 누적 분산의 증가는 5-6번째 주성분 이후에 **둔화**되기 시작한다. 

따라서 이 데이터셋에서 최적의 주성분 수는 **5-6**개로 결정된다.
이것을 토대로 `data`를 압축해 6개의 특성으로 이루어진 `data_pca`를 생성한다.

```python
# 6개의 주성분을 유지하는 PCA 선언 
pca = PCA(n_components=6)

# 기존 data를 pca에 fit_transform
data_pca = pca.fit_transform(data)

# 압축된 데이터 셋 생성
data_pca = pd.DataFrame(data_pca, columns=['PC'+str(i+1) for i in range(pca.n_components_)])

# 인덱스로 빼 두었던 CustomerID 다시 추가
data_pca.index = data.index
```

# K-Means 클러스터링

지정된 그룹 수(**K**)로, 각 군집의 평균(**Means**)을 활용하여 K개의 군집(**cluster**)으로 묶는 방식으로 학습하는 비지도 학습 알고리즘 중 하나.

각 실행마다 클러스터에 다른 레이블을 할당할 수 있다. 즉 처음에 어떤 데이터 포인트를 선택했느냐에 따라 업데이트 방향성이 달라짐. 이를 해결하기 위해 각 클러스터의 샘플 빈도를 기반으로 레이블을 교환하는 추가 단계를 수행한다. 이를 통해 다른 실행에서도 일관된 레이블 할당이 보장된다.

* 클러스터의 샘플 빈도에 기반한 레이블의 교환
1. K-Means 클러스터링 진행 후 임의의 클러스터에 데이터 포인트 할당
2. 데이터 포인트 수 계산
3. 가장 많은 데이터 포인트를 가진 클러스터에 새로운 레이블 할당

> 3번째 단계를 통해 원래의 클러스터링 결과에 해당 새로운 레이블 매핑을 적용해 일관된 레이블을 할당시킬 수 있도록 처리를 해 주었다.

```python
from sklearn.cluster import KMeans
from collections import Counter

# k=3개의 클러스터로 K-Means 클러스터링 적용
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=100, random_state=0)
kmeans.fit(data_pca)

# 각 클러스터의 빈도수 구하기
cluster_frequencies = Counter(kmeans.labels_) 

# 빈도수에 기반하여 이전 레이블에서 새 레이블로의 매핑 생성
label_mapping = {label: new_label for new_label, (label, _) in 
                 enumerate(cluster_frequencies.most_common())}

# 매핑을 적용하여 새 레이블 얻기
new_labels = np.array([label_mapping[label] for label in kmeans.labels_])

# 원래 데이터셋에 새 클러스터 레이블 추가
user_data['cluster'] = new_labels

# PCA 버전의 데이터셋에 새 클러스터 레이블 추가
data_pca['cluster'] = new_labels

# K-Means 분류된 결과 보기
user_data.head()

# 각 군집별로 몇 명의 고객이 있는지 확인
user_data.value_counts('cluster')
```

![각 군집별 고객수 이미지](/assets/img/cluster.png)
_0,1,2 군집별로 각각 2906, 632, 541명을 군집화했다._


# 시각화 및 결과 분석
데이터에서 가장 많은 분산을 포착하는 최상위 주성분 2개를 선택해 시각화하여 클러스터의 분리와 응집의 품질을 시각적으로 확인할 수 있다.
```python
# 각 클러스터 별 데이터 분리 
cluster_0 = data_pca[data_pca['cluster'] == 0]
cluster_1 = data_pca[data_pca['cluster'] == 1]
cluster_2 = data_pca[data_pca['cluster'] == 2]


# 클러스터 별 시각화
plt.scatter(cluster_0['PC1'], cluster_0['PC2'], color = 'orange', alpha = 0.7, label = 'Group1')
plt.scatter(cluster_1['PC1'], cluster_1['PC2'], color = 'red', alpha = 0.7, label = 'Group2')
plt.scatter(cluster_2['PC1'], cluster_2['PC2'], color = 'green', alpha = 0.7, label = 'Group3')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
```

![cluster_2d](/assets/img/cluster_2d.png)

## 3D 시각화
최상위 주성분을 3개를 선택한다면 3D 시각화를 할 수 있다.
```python
# !pip install plotly==5.18.0 -> 라이브러리 설치 필요

# 색상 지정 
colors = ['red', 'blue', 'green']

# 각 클러스터별 데이터 분릴
cluster_0 = data_pca[data_pca['cluster'] == 0]
cluster_1 = data_pca[data_pca['cluster'] == 1]
cluster_2 = data_pca[data_pca['cluster'] == 2]

# 3D Scatter Plot 생성
import plotly.graph_objects as go
fig = go.Figure()

# 각 클러스터별 데이터 표기 
fig.add_trace(go.Scatter3d(x=cluster_0['PC1'], y=cluster_0['PC2'], z=cluster_0['PC3'], 
                           mode='markers', marker=dict(color=colors[0], size=5, opacity=0.4), name='Group 1'))
fig.add_trace(go.Scatter3d(x=cluster_1['PC1'], y=cluster_1['PC2'], z=cluster_1['PC3'], 
                           mode='markers', marker=dict(color=colors[1], size=5, opacity=0.4), name='Group 2'))
fig.add_trace(go.Scatter3d(x=cluster_2['PC1'], y=cluster_2['PC2'], z=cluster_2['PC3'], 
                           mode='markers', marker=dict(color=colors[2], size=5, opacity=0.4), name='Group 3'))

# 범례 및 제목 영역 설정
fig.update_layout(
    title=dict(text='3D Visualization of Customer Clusters in PCA Space', x=0.5),
    scene=dict(
        xaxis=dict(backgroundcolor="grey", gridcolor='white', title='PC1'),
        yaxis=dict(backgroundcolor="grey", gridcolor='white', title='PC2'),
        zaxis=dict(backgroundcolor="grey", gridcolor='white', title='PC3'),
    ),
    width=900,
    height=800
)

fig.show()
```

![cluster_3d](/assets/img/cluster_3d.png)

## 고객 세그먼테이션을 통한 인사이트와 전략
이번 프로젝트를 통해 Recency, Frequency, Monetary 세 가지 기준에 따라 고객을 세그먼테이션하였다. 이 세 가지 기준을 점수화해 고객을 세그먼테이션하여 타겟 마케팅 전략을 세울 수 있다.

예를 들어 R,F,M가 각각 5점인 경우, RFM 점수의 최대값은 15이며 이에 따라 고객 등급이 매겨진다.
- VVIP : 15점
- VIP : 12점
- 골드 : 9~11점
- 실버 : 6~8점
- 브론즈 : 3~5점
- 패밀리 : 1~2점

위와 같은 등급에 따라 각각의 고객 군을 대상으로 다른 할인 혜택과 마케팅 전략을 세울 수 있다.

또는 RFM 총점이 아닌, 각 구성 요소의 점수에 따라 그룹화도 가능하다.

예를 들어 RFM 점수가 15점이면 VVIP로 지정하고, R이 1점, 나머지가 5점인 고객은 잠재 구매력이 굉장히 높지만 최근에 구매를 하지 않은 고객이기 때문에 '구매를 했을 때 추가 할인을 제공하는 프로모션'을 진행해 구매를 유도할 수 있다.

## RFM 분석 & K-Means 클러스터링을 사용한 고객 세그먼테이션
이러한 방식이 갖는 강점:
1. 풍부한 데이터의 사용 : 여러개의 피처들을 그대로 사용할 수 있다
2. 확장성 : 다양한 유형의 데이터를 처리하고, 새로운 피처를 생성하거나 제거하고 싶은 경우 모델 재사용 가능
3. 세밀한 세그멘테이션 : 풍부한 데이터를 사용해 보다 세밀하게 가능
4. 숨겨진 패턴 분석 : ML/DL 기법으로 사람이 쉽게 판별하기 어려운, 혹은 판별하기 복잡한 패턴을 찾을 수 있다.

K-Means 클러스터링을 통해 고객을 총 3개의 그룹으로 세그먼테이션했다.
각 그룹 별 어떤 차이가 있는가 확인해보고 타겟 마케팅 전략을 세워보자.

```python
# 각 클러스터 별로 그룹을 나누어 데이터 생성
group1 = user_data[user_data['cluster'] == 0]
group2 = user_data[user_data['cluster'] == 1]
group3 = user_data[user_data['cluster'] == 2]

# 각 그룹별 요약 정보 확인
group1.describe()
group2.describe()
group3.describe()
```
![group_desc](/assets/img/group_desc.png)
_군집화한 고객 별로 가장 효과적인 마케팅 전략을 세워보자._