---
title: "[NLP] Language Model using n-gram Tutorial - Part 3"
categories:
  - snlp
tags:
  - ngram
  - nlp
  - statistical
  - language model
  - smoothing
sitemap: true
toc: true
toc_label: "Table of Contents"
toc_sticky: true
---

> 본 포스트 시리즈는 FSNLP(Foundation of Statistical Natural Language Processing)의 내용 중에서,
language modeling, HMM(hidden markov, model)을 이용한 POS(part-of-speech) Tagger 구현에 관한 
내용을 정리하는 것을 목적으로 합니다.

[지난 포스트][previous post]에서는 nltk에서 제공하는 `gutenberg corpus`로 구성한 `n-gram` 모델로
[MLE]({% post_url 2018-10-18-probability-concept %}) 를 계산하는 방법과 **Data Sparseness** 문제점을 알아보았습니다.

이번 시간에는 이런 문제점을 극복하기 위한 방법을 알아보겠습니다.

## Problem Definition of MLE

`training set`에 아무리 많은 단어가 있다 할지라도, `training set`에서 발견되지 않은 데이터(unseen data)가 발생할 수 있고, 
해당 `n-gram`에 대해 확률을 구할 수가 없다. 이 문제는 또한 문장에 대한 확률 $$P(w_1 \cdots w_n)$$ 을 계산하는데 전파됩니다. 
따라서, `training set`에 존재하지 않는 단어 및 `n-gram`에 대해, 0이 아닌 확률을 부여할 수 있는 방법(smoothing)이 필요하게 됩니다.

## Smoothing 

### Laplace's law

가장 오래된 해결 방법으로 Laplace's law를 적용해 볼 수 있습니다. `adding one` 기법으로도 알려진 이 방법은 `unseen data`에 작은 확률을
부여하는 방법으로 다음과 같이 정의됩니다. 아래 수식에서의 `B`는 training instance가 속할 수 있는 `bin(class)`의 개수를 의미합니다.

$$ P_{Lap}(w_1 \cdots w_n) = \frac{C(w_1 \cdots w_n)+1}{N+B} $$

하지만, `Laplace's law`는 `vocabulary size`에 의존적인 특징으로 인해, `training set`이 충분하지 않을 때, `unseen` 데이터에 지나치게 많은
확률을 부여하는 단점이 있습니다.   
 
$r$번 출현한 `n-gram`에 대한 우도는 $ (r+1) / (N+B) $ 이므로, `n-gram`의 `expected frequency` $f_{Lap}$ 는 
$\frac{N(r+1)}{(N+B)}$ 로 추정할 수 있습니다.

### Lidstone's law and Jeffreys-Perks law

`Laplace's law`의 문제점을 해결하기 위해 널리 사용되는 방법은 `Lidstone's law`입니다. 1을 더하는 대신, positive value로 $ \lambda $를
더하는 것으로 다음과 같이 정의됩니다.

$$ P_{Lid}(w_1 \cdots w_n) = \frac{C(w_1 \cdots w_n) + \lambda}{N + B\lambda} $$

가장 널리 사용되는 $ \lambda $ 는 0.5인데, MLE에 의해 최대화된 값과 동일한 양을 사용하게 되어 이론적으로 합리적으로 보이기 때문이다.
이와 같이 $ \lambda=0.5 $로 설정한 경우, `Jeffreys-Perks law` 혹은 `ELE(Expectation Likelihood Estimation)`이라고 합니다.

`Lidstone's law` 혹은 `ELE`는 $\lambda$ 를 작은 값으로 설정하여 `unseen data`에 지나치게 많은 확률을 부여하는 것을 방지할 수 있습니다만, 
다음의 두 가지의 단점이 남아있습니다.
 1. 좋은 $ \lambda $ 를 찾기 위한 방법이 필요하다.
 2. 낮은 빈도수에서는 실제 데이터와 잘 부합하지 않는다.(`Lidstone's`는 MLE 추정치를 항상 선형으로 감소시키므로)  

### Good Turing

#### 정의

> **wikipidea** :Good–Turing frequency estimation was developed by [Alan Turing]([https://en.wikipedia.org/wiki/Alan_Turing]) 
and his assistant [I. J. Good]([https://en.wikipedia.org/wiki/I._J._Good]) as part of their efforts at Bletchley Park to crack
German ciphers for the Enigma machine during World War II. Turing at first modelled the frequencies as a multinomial distribution,
but found it inaccurate. Good developed smoothing algorithms to improve the estimator's accuracy.

Good Turing은 한번도 발견되지 않은 n-gram의 빈도는 한번 출현한 n-gram의 빈도로 추정할 수 있다는 것으로, 아래의 식으로 빈도수를 조정합니다. 

$$
\begin{align*}
P_{GT} & = r^\star / N \tag{1} \label{equa_1}\\
r^\star & = (r+1)\frac{E(N_{r+1})}{E(N_r)} \tag{2} \label{equa_2} 
\end{align*}
$$

|Term|Definition|
|----|----------|
|$N$ |Number of Training Instance|
|$r$|Frequency(or Rate) of an $n$-gram|
|$r^\star$|Adjusted Frequency|
|$N_r$|Number of classes that have $r$ training instances in them|

#### 직관적인 설명

빈도수 r을 갖는 n-gram의 확률은 $r/N$으로 구할 수 있습니다.
이 것은 이전 포스트에서 살펴본 것처럼 출현한 n-gram들에 대한 확률을 최대화하기(MLE) 때문에, unseend data를 위한 확률 공간이 존재하지 않게 됩니다. 
따라서, unseen data를 위한 공간을 확보하기 위해 $r\neq0$인 n-gram 들의 빈도를 낮추고 unseen data를 위한 공간(Frequency)을 확보하는 것입니다. 
일반적으로 $r+1$의 확률 공간은 $r$에 비해 상대적으로 작을 것이기 때문에, $r+1$이 차지하는 공간을 $r$에 해당하는 n-gram에 동등하게 분할하여 
사용하는 것이 이 방법의 주요한 개념입니다.

그렇다면 $r$일 때 차지하는 확률 공간은 얼마나 될까요? $r$번 출현한 n-gram의 수는 $N_r$이므로, 다음과 같이 정의할 수 있습니다.

$\text{Space}(r) = r \times N_r$  
$\text{Overall Space} = \sum_{r=1}^{n} rN_r$  
$\text{Space ratio} = \frac{rN_r}{\sum_{r=1}^{n} rN_r}=\frac{rN_r}{N}$

따라서, 빈도수 r+1에 해당하는 n-gram들이 모든 인스턴스에서 차지하는 비율, 즉, r+1의 확률은 $\frac{(r+1)N_{r+1}}{N}$으로 얻어지고, 
이 확률 공간을 $N_r$에 동등하게 할당해서 빈도수 r에서의 각 n-gram이 갖는 확률을 추정하게 되면, 처음에 정의한 $P_{GT}$와 같은 수식을 얻게 됩니다. 

처음에 정의한 수식에서, training set에서 r번 출현한 단어는 (\ref{equa_2})에 따라 $r^\star$로 조정되고, (\ref{equa_1})에 따라 전체 수로 나누어 
개별적인 n-gram이 출현할 확률($P_{GT}$)을 구하는 과정입니다.

#### 문제점과 해결책

지금까지 알아본 Good-Turing에 의한 보정은 높은 빈도수를 갖는 n-gram에서는 잘 동작하지 않습니다. 예를 들어, $N_{r+1} = 0$인 경우에, $N_r$의 확률을
구할 수 없기 때문입니다. 이 문제는 다음의 두 가지 방법으로 해결합니다.

* **Solution 1**: 낮은 빈도수의 (r < k for some constant k)에 대해서 $P_{GT}$를 적용하고, 높은 빈도수에서는 MLE에 의한 추정을 적용 

* **Solution 2**: $(r, Nr)$에 대해 smoothing function *S*를적용하여 $E(N_r)$에 적용

두 가지 방법을 적용하여, 다음의 개선된 estimator를 정의합니다.

**Good-Turing Estimator**

If $ C(w_1 \cdots w_n) = r \gt 0 $,
$$ P_{GT}(w_1 \cdots\ w_n) = \frac{r^{\star}}{N}\ \text{where}\ r^{\star}=\frac{(r+1)S(r+1)}{S(r)}$$

if $C(w_1 \cdots w_2) = 0 $,  
$$P_{GT}(w_1 \cdots w_n) = \frac{1-\sum_{r=1}^{\infty N_r\frac{r^\star}{N}}}{N_0} = \frac{N_1}{N} \cdot \frac{1}{N_0}$$

#### Re-normalization

Solution 1과 Solution 2 혹은 어떠한 복잡한 방법을 적용하더라도, 확률 분포가 잘 구성되도록 정규화해야합니다.
(i) 정규화하는 방법은 간단히 전체 n-gram에 대해 전체 합이 1이 되도록 정규화하는 방법과 (ii) unseen data에 할당한 확률 분포는 유지하고 나머지에
대해서만 정규화하는 방법이 있습니다. 예를 들어, Good-Turing에 의해 unseen data에 할당된 확률이 1/20이라면, seen data에 할당된 확률의 총합이
19/20이 되도록 정규화합니다.

#### Bigram Example

**Good-Turing Estimator**와 **Re-normalization**을 bi-gram 모델에 적용하는 예제를 살펴보겠습니다.

1. Vocabulary = {a, b, c}
  
2. Number of possible bi-grams = $3^2$ = 9

3. Observed bi-grams = {ba, ab, ba, aa, ac, cb, bc, ca, ac, ca, ac}
 * ba: 2, ab: 1, aa:1, ac: 3, cb:1, bc: 1, ca: 2
 * $N_0=2, N_1=4, N_2=2, N_3=1, \sum_0^3 N_i = N = 11 $
 
4. Apply GT up to r < k for k = 3
 * $P_{GT}(r=0) = (N_1)/(N \cdot N_0) = 2/11 $ 
 * $P_{GT}(r=1) = 2(N_2)/(N \cdot N_1) = 1/11 $  
 * $P_{GT}(r=2) = 3(N_3)/(N \cdot N_2) = 3/22 $  
 * $P_{MLE}(r=3) = 3/11 $  

5. Re-normalize
 * $1 - P_{GT}(r=0) \times N_0 = 7/11$
 * $\text{Constant Value} \times \sum_{i=1}^{3} {P_{GT}(r=i) \times N_i} = 7/11$
 * $\text{Constant Value} = \frac{7/11}{\sum_{i=1}^{3} P_{GT}(r=i) \times N_i}=\frac{7/11}{10/11}=7/10$
 * Finally,
  * $\text{constant value} \times P_{GT}(r=1) = 14/110 $ 
  * $\text{constant value} \times P_{GT}(r=2) = 7/110 $
  * $\text{constant value} \times P_{GT}(r=3) = 21/220 $
 * validataion
  * $\sum_{i=0}^{3} P_{GT}(r=i) \times N_i = 2 \cdot 2/11 + 4 \cdot 14/110 + 2 \cdot 7/110 + 3 \cdot 21/220 = $

## Interpolation

[previous post]:https://sept1022.github.io/snlp/2018-10-18-language-model-part-2/
[MLE post]:http://sep1022.github.io/snlp/