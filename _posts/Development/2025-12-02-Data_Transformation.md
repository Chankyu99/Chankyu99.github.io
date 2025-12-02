---
title: "데이터 변환(Data Transformation)"
description: "Pandas로 연봉 데이터 다루기"
date: 2025-12-02
categories: [Development, Data Science]
tags: [Pandas, Data Cleaning, EDA]
math: false
---

오늘은 데이터 변환 노드 학습을 진행하며 마주친 Pandas의 핵심 전처리 메서드들을 정리했다.
특히 헷갈리기 쉬운 **병합(Merge/Join)**, **구조 변경(Pivot/Melt)**, 그리고 **스케일링(Scaling)** 기법을 위주로 정리했다.

---

## 1. 데이터 합치기

두 개의 데이터프레임을 하나로 합칠 때 사용하는 메서드다.
가장 큰 차이는 **"무엇을 기준으로 합치는가"**이다.

### `merge()`
**열(Column) 값**을 기준으로 병합한다. (SQL의 JOIN과 유사)

* `on`: 기준이 되는 컬럼 지정 (양쪽 이름이 다르면 `left_on`, `right_on` 사용)
* `how`: 병합 방식 (`inner`, `outer`, `left`, `right`)

```python
# df_a의 'column1'와 df_b의 'column2'를 기준으로 모든 데이터 포함(Outer Join)
df_a.merge(df_b, left_on='column1', right_on='column2', how='outer')
```

### `join()`

\*\*인덱스(Index)\*\*를 기준으로 병합한다.

  * 기본적으로 `how='left'`가 적용된다.
  * 컬럼명이 겹칠 경우 `lsuffix`, `rsuffix`로 접미사를 붙여 구분한다.

<!-- end list -->

```python
# 인덱스를 기준으로 합치되, 겹치는 컬럼명에 '_a', '_b' 꼬리표 부착
df_a.join(df_b, lsuffix='_a', rsuffix='_b')
```

---

## 2\. 데이터 연결하기

단순히 데이터를 위아래로 이어 붙일 때는 `pd.concat()`을 사용한다. 이때 **인덱스 중복** 문제가 발생할 수 있다.
`pd.concat()`은 기존 인덱스를 그대로 유지하므로, 합친 후 인덱스 0이 여러 개 존재할 수 있습니다. (`loc[0]` 조회 시 여러 행 반환).

> **해결책:** `ignore_index=True`를 쓰거나, 후처리로 `reset_index`를 해야 한다.

```python
# 방법 1: 합칠 때 인덱스 재설정
pd.concat([df1, df2], ignore_index=True)

# 방법 2: 합친 후 인덱스 초기화 (기존 인덱스는 삭제)
df_merged.reset_index(drop=True, inplace=True)
```

---

## 3\. 데이터 그룹화 및 집계 (Groupby & Aggregation)

데이터를 특정 기준(범주)에 따라 묶고, 통계량(평균, 합계 등)을 계산할 때 사용한다.

### `groupby()`

특정 컬럼을 기준으로 데이터를 그룹핑한다. 이때 **수치형 컬럼만 선택**하거나 `numeric_only=True` 옵션을 줘야 에러(TypeError)를 방지할 수 있다.

```python
# 'Gender'별로 묶어서 'Salary'의 평균 구하기
salary_df.groupby('Gender')['Salary'].mean()
```

### 다중 그룹핑 및 인덱스 리셋

여러 기준을 적용하거나, 결과를 다시 깔끔한 데이터프레임 형태로 만들 때 사용한다.

  * `reset_index()`: 그룹핑으로 인해 인덱스로 들어간 컬럼들을 다시 데이터 열로 복구시킨다.

<!-- end list -->

```python
# 성별(Gender)과 국가(Country)별 연봉 평균을 구하고, 새로운 인덱스 부여
avg_df = salary_df.groupby(['Gender', 'Country'])['Salary'].mean().reset_index()
```

### `agg()`

여러 가지 통계 함수를 한 번에 적용할 때 유용하다.

```python
# 연봉의 평균(mean)과 합계(sum)를 동시에 확인
salary_df.groupby('Gender')['Salary'].agg(['mean', 'sum'])
```

---

## 4\. 데이터 구조 변경

### `pivot()`: Long → Wide (요약표 만들기)

  * 세로로 긴 데이터를 가로로 넓게 펼쳐서 보기 좋게 만든다.
  * 행(`index`), 열(`columns`), 값(`values`)을 지정합니다.

<!-- end list -->

```python
# 회사(row)와 분기(col)별 매출(value) 표 만들기
pd.pivot(sales_df, index='company', columns='quarter', values='sales')
```

### `melt()`: Wide → Long (분석용 데이터 만들기)

  * 가로로 넓은 데이터를 세로로 길게 녹여서(Melt) 컴퓨터가 학습하기 좋은 형태로 만든다.
  * 고정할 기준 열(`id_vars`)을 제외한 나머지 열들을 행으로 내립니다.

<!-- end list -->

```python
# company를 기준으로 나머지 분기 컬럼들을 행으로 변환
pd.melt(df, id_vars='company', var_name='quarter', value_name='sales')
```

---

## 5\. 범주형 데이터 처리: One-Hot Encoding

### `get_dummies()`

범주형 데이터를 0과 1로 변환한다.

> `drop_first=True` : N개의 범주를 표현할 때 첫 번째 컬럼을 삭제하여 N-1개만 사용한다. (다중공선성 문제 방지)

```python
pd.get_dummies(df, columns=['Gender', 'Country'], drop_first=True)
```

---

## 6\. 수치형 데이터 스케일링 (Scaling)

Scaling에 대해 수식을 포함한 자세한 내용은 [여기](https://www.google.com/search?q=https://chankyu99.github.io/posts/feature_scaling/)에서 확인할 수 있다.
