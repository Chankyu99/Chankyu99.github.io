---
title : "무지성 plt.plot()만 쓰는 너에게 - pyplot 뜯어보기"
description : "Matplotlib.pyplot으로 알아보는 파이썬의 객체 지향"
date : 2025-12-19
categories : [Development, Data Science]
tags : [Matplotlib, Seaborn]
pin : false
math : true
mermaid : true
---

# Matplotlib.pyplot

데이터 시각화에서 빼놓을 수 없는 라이브러리, 바로 `Matplotlib.pyplot`이다.

<img src="/assets/img/plt.png" width="700" height="700"/>
_이름이 너무 길어요_

가장 많이 쓰는 라이브러리 이지만, 막상 라이브러리를 사용할 때마다 공식문서를 들락날락 거리게하는 아주 귀찮은 라이브러리라고 할 수 있겠다.

그럼 이녀석이 왜 이렇게 귀찮은 라이브러리가 되었을까? 그 이유는 라이브러리의 구조 때문이다.

우선 라이브러리명(`Matplotlib.pyplot`)에 주목하자. 

오늘 뜯어볼 녀석은 바로 `pyplot.py` 파일이다. [링크](https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/pyplot.py)를 통해 코드를 직접 확인 해볼 수 있다.

![2](/assets/img/plt2.gif)
_음.. 이걸 다보라고?_

물론 필요한 부분만 볼 것이다. 우리가 주목할 부분은 `figure()`, `axes()`, 그리고 `plot()`함수이다.

```python
# figure()
def figure(num=None, ...):
    # ... (생략)
    if manager is None:
        # 종이가 없다면? 만들어! (figure 객체 생성)
        manager = new_figure_manager(num, ...) 
    return manager.canvas.figure

# axes()
def axes(arg=None, **kwargs):
    fig = gcf()  # 현재 도화지(Figure 객체)를 가져옴 (Get Current Figure)
    # ...
    return fig.add_axes(arg, **kwargs) # 그 Figure 객체의 메서드(add_axes)를 실행

# plot() -> 우리가 아는 plt.plot() 맞음 ㅇㅇ
def plot(*args, **kwargs):
    return gca().plot(*args, **kwargs)
```

코드를 통해서 다음과 같은 사실을 알 수 있다.

`figure()`

- `Figure` 객체가 이미 존재하면 가져오고, 없으면 새로 생성한다. (유식한 말로 **Get or Create**)
- `Figure`라는 **클래스**를 사용하기 떄문에, `fig` 변수에 해당 클래스의 인스턴스를 얻게 되었다고 생각하면 된다.

`axes()`

- `Figure` 객체의 메서드(`add_axes`)를 실행하여 `Axes` 객체를 생성하고 반환한다.
- `Figure` 객체가 다스리는 녀석이 된다.

`plot()`

- `Axes` 객체의 메서드(`plot`)를 실행하여 그래프를 그린다.
- `pyplot`에는 `plot`기능이 없지만, `Figure`와 `axes` 객체가 생성이 되었으므로, 작동하기만 하면 `plot` 기능이 가능하다.
- 객체 지향의 **캡슐화**를 잘 보여주는 예시라고 할 수 있다.

결론을 말하자면, `Matplotlib.pyplot`의 `fig`, `ax`를 사용하는 것이 파이썬의 대표적인 특징인 **<mark>객체 지향</mark>**을 잘 보여주는 것이라고 할 수 있다.

객체 지향의 특징으로 `Matplotlib.pyplot`을 구현한다면 다음과 같은 특징을 알 수 있다.

- `Axes` 객체를 통해 여러 그래프(`Subplot`)을 그리는데 있어서 **직관적인 명시**가 가능하다.
- **API 기반의 시각화**로 코드를 통해 인자값이나 주변값을 설정해 **다양한 커스터마이징**이 가능하다.

# 결론

흔한 경우처럼 `plt.plot()`을 쓰는 것이 편하겠지만, `Subplot`을 다룰경우 특정 객체를 지향하고 그것을 사용하는 것이 `Matplotlib.pyplot`의 **<mark>객체 지향의 특징</mark>**을 잘 보여주는 것이라고 할 수 있다. 여러가지 데이터의 실습을 통해서 익숙해지는 것이 데이터 시각화에 있어서 훨씬 더 정교하게 제어가 가능하기에 알아두는 것이 좋다고 생각한다.

# 📚 참고자료
[1] [Matplotlib.pyplot 공식 문서](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html)

[2] [Matplotlib.pyplot Github](https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/pyplot.py)

