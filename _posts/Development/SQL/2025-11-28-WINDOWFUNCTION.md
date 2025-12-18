---
title: "SQL 실무에 많이 쓰이는 Window Function"
description: "SQL 심화 - SQL로 데이터 분석하기"
date: 2025-11-28 
categories: [Development, SQL]
tags: [SQL, Window Function]
math: true
# pin: true
# 게시글 썸네일
# image :
# path : /commons/devices-mockup.png
---

개인적으로 SQL을 배우면서 쿼리문 작성에 활용하기가 가장 어려웠던 부분이었다.
정리를 통해 개념을 확실히 다져보고자 한다.

---

# WINDOW FUNCTION? 
**순위, 집계 등 행과 행 사이의 관계를 정의하는 함수**.


## 순위 함수 : RANK(), DENSE_RANK(), ROW_NUMBER()

```sql
SELECT column_name,
RANK() OVER (PARTITION BY column_name ORDER BY column_name WINDOWING)
DENSE_RANK() OVER (PARTITION BY column_name ORDER BY column_name WINDOWING)
ROW_NUMBER() OVER (PARTITION BY column_name ORDER BY column_name WINDOWING)
FROM table_name;
```

**순위 함수는 ()에 들어가는 인수(ARGUMENT)를 쓰지 않는다!**

- RANK() : 공동 등수 인정 O, 다음 등수 건너뜀 (1,1,3,4 ...)
- DENSE_RANK() : 공동 등수 인정 O, 다음 등수 안건너뜀 (1,1,2,3...)
- ROW_NUMBER() : 공동 등수 인정 X, Primary Key 오름차순 (1,2,3,4 ...)

## 집계 함수 : SUM(), AVG(), MAX(), MIN()
> GROUP BY 없이도 "내 옆에" 집계 결과를 붙일 수 있다! (OVER)

```sql
SELECT column_name,
SUM(column_name) OVER (PARTITION BY column_name ORDER BY column_name WINDOWING)
AVG(column_name) OVER (PARTITION BY column_name ORDER BY column_name WINDOWING)
MAX(column_name) OVER (PARTITION BY column_name ORDER BY column_name WINDOWING)
MIN(column_name) OVER (PARTITION BY column_name ORDER BY column_name WINDOWING) 
FROM table_name
```

- SUM(컬럼명) : 컬럼을 기준으로 합계 계산
- AVG(컬럼명) : 컬럼을 기준으로 평균 계산
- MAX(컬럼명) : 컬럼을 기준으로 최대값 계산
- MIN(컬럼명) : 컬럼을 기준으로 최소값 계산

> 집계 함수는 ()에 들어가는 인수(ARGUMENT)를 반드시 써주어야 한다!

## 행 순서 함수 : FIRST_VALUE(), LAST_VALUE(), LAG(), LEAD()

```sql
SELECT column_name,
FIRST_VALUE(column_name) OVER (PARTITION BY column_name ORDER BY column_name ROWS BETWEEN A AND B) 
LAST_VALUE(column_name) OVER (PARTITION BY column_name ORDER BY column_name ROWS BETWEEN A AND B) 
LAG(column_name, n) OVER (PARTITION BY column_name ORDER BY column_name ROWS BETWEEN A AND B) 
LEAD(column_name, n) OVER (PARTITION BY column_name ORDER BY column_name ROWS BETWEEN A AND B) 
FROM table_name
```
- FIRST_VALUE(column_name) : 조건을 만족하는 가장 먼저 나온 값을 구한다
- LAST_VALUE(column_name) : 조건을 만족하는 가장 마지막 나온 값을 구한다
- LAG(column_name, n) : 현재 행의 n번째 이전 행의 값을 구한다
- LEAD(column_name, n) : 현재 행의 n번째 다음 행의 값을 구한다

> WINDOWING 절이 꼭 필요하므로 알아두자. 

- CURRENT ROW : 현재 행
- UNBOUNDED PRECEDING : 윈도우의 시작 위치가 첫번째 행
- UNBOUNDED FOLLOWING : 윈도우의 마지막 위치가 마지막 행

## 그룹 함수 : ROLLUP(), CUBE(), GROUPING SETS()

```sql
SELECT column_name,
ROLLUP(column_name) OVER (PARTITION BY column_name ORDER BY column_name WINDOWING) 
CUBE(column_name) OVER (PARTITION BY column_name ORDER BY column_name WINDOWING)
GROUPING SETS(column_name) OVER (PARTITION BY column_name ORDER BY column_name WINDOWING)
FROM table_name
```
- ROLLUP(column_name) : 계층적인 소계 (예 : 연도별 > 월별 > 일별)
- CUBE(column_name) : 가능한 모든 조합의 소계 
- GROUPING SETS(column_name) : 내가 지정한 조합의 소계

