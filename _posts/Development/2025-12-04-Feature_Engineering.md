---
title: "특성 공학(Feature Engineering)"
description: "titanic 데이터를 이용한 특성 공학"
date: 2025-12-02
categories: [Development, Data Science]
tags: [Pandas, Data Cleaning, Feature Engineering]
math: false
---

오늘은 Kaggle에서 제공하는 `titanic` 데이터셋을 이용해 특성 공학을 알아보았다.

---

# 1. 데이터 확인

사용할 데이터가 어떻게 구성되어있는지 확인하는 것이 중요하다.
`data.info()`를 통해 구조를 확인해보자.

```python
data.info()
```

![info](/assets/img/info.png)

---

# 2. 데이터 전처리

## 결측치 처리

결측치를 처리해야할 컬럼은 총 3가지였다.

- `age` : 결측치를 **중간값**으로 대체
- `embarked` : 결측치를 **최빈값**으로 대체
- `deck` : 결측치가 너무 많아 컬럼 자체를 **삭제**

그 외에 불필요한 컬럼 역시 **제거**하였다.

## 특성 공학
특성 공학을 통해 여기에선 여러가지 컬럼을 생성해 보았다.

```python
# 'Family_size' 컬럼 추가
data['Family_Size'] = data['sibsp'] + data['parch'] + 1

# '1인당 요금(Fare_Per_Person)' 컬럼 추가
data['Fare_Per_Person'] = data['fare'] / data['Family_Size'] # 전체 요금을 가족 수로 나누기

# 나이대 구분 (Age_group) 컬럼 추가
data['Age_Group'] = pd.cut(data['age'],bins = [0,10,20,60,100],labels = ['child','Teen','Adult','Senior'])
data = pd.get_dummies(data, columns=['Age_Group'], drop_first=True)

# 불리언(True/False) 컬럼만 찾아서 정수(1/0)로 변경
bool_cols = data.select_dtypes(include=['bool']).columns
data[bool_cols] = data[bool_cols].astype(int)
```

- `family_size` : `sibsp`와 `parch`를 합친 컬럼
- `fare_per_person` : `fare`를 `family_size`로 나눈 컬럼
- `Age_group` : `age`를 `child`, `teen`, `adult`, `senior`로 구분한 컬럼

이후 생성된 컬럼들을 비롯해 전체 컬럼에서 범주형 데이터들에 대해 인코딩을 진행해주었다.

![info](/assets/img/info2.png)

---

# 3. 모델링 후 비교 분석

로지스틱회귀 모델을 이용해 간단하게 예측 정확도를 구하고 원본 데이터와 비교해 보았다.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 원본 데이터 준비
df_origin = sns.load_dataset('titanic')
df_origin.dropna(subset=['age', 'embarked'], inplace=True) # 결측치 제거
df_origin = pd.get_dummies(df_origin, columns=['sex', 'embarked'], drop_first=True)

# 피쳐와 타깃 분리
X_origin = df_origin[['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male', 'embarked_Q', 'embarked_S']]
y_origin = df_origin['survived']

# 학습과 테스트를 위한 데이터셋 분리
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_origin, y_origin, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 준비
model = LogisticRegression(max_iter=1000)

# 모델 학습 및 예측
model.fit(X_train_o, y_train_o)
pred_o = model.predict(X_test_o)
# 예측 결과와 테스트 정답값을 비교해서 정확도 구하기
accuracy_o = accuracy_score(y_test_o, pred_o)
print("원본 데이터 정확도 : ", accuracy_o)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 특성 공학으로 만든 데이터 준비
df_origin = data

# 피쳐와 타깃 분리
X_origin = df_origin[['pclass', 'age', 'sibsp', 'parch', 'fare', 'alone',
       'sex_male', 'embarked_Q', 'embarked_S', 'Family_Size',
       'Fare_Per_Person', 'Age_Group_Teen', 'Age_Group_Adult',
       'Age_Group_Senior']]
y_origin = df_origin['survived']

# 학습과 테스트를 위한 데이터셋 분리
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_origin, y_origin, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 준비
model = LogisticRegression(max_iter=1000)

# 모델 학습 및 예측
model.fit(X_train_e, y_train_e)
pred_e = model.predict(X_test_e)

# 예측 결과와 테스트 정답값을 비교해서 정확도 구하기
accuracy_e = accuracy_score(y_test_e, pred_e)
print("데이터 정확도 : ", accuracy_e)
```

---

# 4. 결과

정확도 상승 폭은 **0.0017%로 매우 미세한 차이**를 보였다. <br>
왜 그럴까 간단하게 생각해 보았는데, 다음과 같은 문제의 가능성을 생각해보았다.

- 데이터 표본 수가 **매우 작음**
- 로지스틱회귀가 해당 데이터에 적용할 수 있는 **적절한 모델**이 아님
- 특성 공학을 통해 생성된 `family size`가 기존 변수와의 **높은 상관관계**를 가지면서 **다중공선성** 문제가 발생함.