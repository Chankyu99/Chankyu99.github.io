---
title: "그리디 알고리즘"
date: 2025-12-18 
categories: [Development, CS & Algorithm]
tags: [Algorithm]
pin: false
math: true
mermaid: true
---

# 실전 문제 풀이 : 큰 수의 법칙

다양한 수로 이루어진 배열이 있을 때, 주어진 수들을 M번 더하여 가장 큰 수를 만드는 법칙이다. 

- 배열의 특정한 인덱스(번호)에 해당하는 수가 연속해서 K번을 초과하여 더해질 수 없다.
- 서로 다른 인덱스에 해당하는 수가 같은 경우에도 서로 다른 것으로 간주한다.

배열의 크기 N, 숫자가 더해지는 횟수 M, 그리고 K가 주어질 때 큰 수의 법칙에 따른 결과를 출력하라.

```python
N, M, K = map(int, input().split())

array = list(map(int, input().split()))

array.sort()

first = array[N - 1]
second = array[N - 2]

result = 0

while True:
    for i in range(K):
        if M == 0:
            break
        result += first
        M -= 1
    if M == 0:
        break
    result += second
    M -= 1

print(result)
```