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

[지난 포스트][previous post]에서는 nltk에서 제공하는 `gutenberg corpus`로 구성한 `n-gram` 모델로
[MLE]({% post_url 2018-10-18-probability-concept %}) 를 계산하는 방법과 **Data Sparseness** 문제점을 알아보았습니다.

이번 시간에는 이런 문제점을 극복하기 위한 방법을 알아보겠습니다.

## 1. Problem Definition

`training set`에 아무리 많은 단어가 있다 할지라도, `training set`에서 발견되지 않은 데이터(`unseen data`)가 발생할 수 있고, 
해당 `n-gram`에 대해 확률을 구할 수가 없게 됩니다. 이 문제는 문장의 확률 $$P(w_1 \cdots w_n)$$ 을 계산하는데 전파됩니다. 
따라서, `training set`에 존재하지 않는 단어 및 `n-gram`에 대해서도 확률을 부여할 수 있는 방법이 필요하게 됩니다.

## 2. Smoothing 

### 2.1 Laplace's law

가장 단순 방법으로 Laplace's law를 적용해 볼 수 있습니다. `adding one` 기법으로도 알려진 이 방법은 `unseen data`에 작은 확률을
부여하는 방법으로 다음과 같이 정의됩니다. 아래 수식에서의 `B`는 training instance가 속할 수 있는 `bin(class)`의 개수를 의미합니다.

$$ P_{Lap}(w_1 \cdots w_n) = \frac{C(w_1 \cdots w_n)+1}{N+B} $$

$r$번 출현한 `n-gram`에 대한 우도는 $ (r+1) / (N+B) $ 이므로, `n-gram`의 `expected frequency` $f_{Lap}$ 는 
$\frac{N(r+1)}{(N+B)}$ 로 추정할 수 있습니다. 

하지만, `Laplace's law`는 `vocabulary size`에 의존적인 특징으로 인해, `training set`이 충분하지 않다면 `unseen data`에 지나치게 많은
확률을 부여하는 단점이 있습니다. 

### 2.2 Lidstone's law and Jeffreys-Perks law

`Laplace's law`의 문제점을 해결하기 위해 널리 사용되는 방법은 `Lidstone's law`입니다. 특정한 수를 더하는 대신, positive value로 $ \lambda $를
더하는 것으로 다음과 같이 정의됩니다.

$$ P_{Lid}(w_1 \cdots w_n) = \frac{C(w_1 \cdots w_n) + \lambda}{N + B\lambda} $$

MLE에 의해 최대화된 값과 동일한 양을 사용하게 되어 합리적일 수 있다는 이유로 $ \lambda $의 값으로 0.5가 널리 사용되는데,  
이와 같이 $ \lambda=0.5 $로 설정한 경우, `Jeffreys-Perks law` 혹은 `ELE(Expectation Likelihood Estimation)`이라고 합니다.

`Lidstone's law` 혹은 `ELE`는 $\lambda$ 를 작은 값으로 설정하여 `unseen data`에 지나치게 많은 확률을 부여하는 것을 방지할 수 있습니다만, 
다음의 두 가지의 어려운 점이 남아있습니다.  
 1. 좋은 $ \lambda $ 를 찾기 위한 방법이 필요하다.
 2. 낮은 빈도수에서는 실제 데이터와 잘 부합하지 않는다.(`Lidstone's`는 MLE 추정치를 항상 선형으로 감소시키므로)
 
### 2.3 Good Turing

#### 2.3.1 정의

**wikipidea** :Good–Turing frequency estimation was developed by [Alan Turing]([https://en.wikipedia.org/wiki/Alan_Turing]) 
and his assistant [I. J. Good]([https://en.wikipedia.org/wiki/I._J._Good]) as part of their efforts at Bletchley Park to crack
German ciphers for the Enigma machine during World War II. Turing at first modelled the frequencies as a multinomial distribution,
but found it inaccurate. Good developed smoothing algorithms to improve the estimator's accuracy.
{: .notice--info}

`Good-Turing`은 한번도 발견되지 않은 `n-gram`의 빈도는 한 번 출현한 `n-gram`의 빈도로 추정할 수 있다는 것으로, 아래의 식으로 빈도수를 조정합니다. 

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

#### 2.3.2 직관적인 설명

빈도수 r을 갖는 `n-gram`의 확률은 $r/N$으로 구할 수 있습니다.
이 것은 이전 포스트에서 살펴본 것처럼 출현한 `n-gram`들에 대한 확률을 최대화하기(`MLE`) 때문에, `unseen data`를 위한 확률 공간이 존재하지 않게 됩니다. 
보통의 절하 방법은 `unseen data`를 위한 공간을 확보하기 위해 $r\neq0$인 `n-gram` 들의 빈도를 낮추고 unseen data를 위한 공간을 확보하는 것입니다. 
일반적으로 $r+1$의 확률 공간은 $r$에 비해 상대적으로 작을 것이기 때문에, $r+1$이 차지하는 공간을 $r$에 해당하는 n-gram에 동등하게 분할하여 
사용하는 것이 이 방법의 주요한 내용입니다.

그렇다면 $r$일 때 차지하는 확률 공간은 얼마나 될까요? $r$번 출현한 n-gram의 수는 $N_r$이므로, 다음과 같이 정의할 수 있습니다.

$\text{Space}(r) = r \times N_r$  
$\text{Overall Space} = \sum_{r=1}^{n} rN_r$  
$\text{Space ratio} = \frac{rN_r}{\sum_{r=1}^{n} rN_r}=\frac{rN_r}{N}$  

따라서, 빈도수 r+1에 해당하는 n-gram들이 모든 인스턴스에서 차지하는 비율, 즉, r+1의 확률은 $\frac{(r+1)N_{r+1}}{N}$으로 얻어지고, 
이 확률 공간을 $N_r$에 동등하게 할당해서 빈도수 r에서의 각 n-gram이 갖는 확률을 추정하게 되면, 처음에 정의한 $P_{GT}$와 같은 수식을 얻게 됩니다. 

처음에 정의한 수식에서, training set에서 r번 출현한 단어는 (\ref{equa_2})에 따라 $r^\star$로 조정되고, (\ref{equa_1})에 따라 전체 수로 나누어 
개별적인 n-gram이 출현할 확률($P_{GT}$)을 구하는 과정입니다.

#### 2.3.3 문제점과 해결책

지금까지 알아본 Good-Turing에 의한 보정은 높은 빈도수를 갖는 n-gram에서는 잘 동작하지 않습니다. $N_{r+1} = 0$인 경우에, $N_r$의 확률을
구할 수 없기 때문입니다. 이 문제는 다음의 두 가지 방법으로 해결합니다.

* **Solution 1**: 낮은 빈도수의 (r < k for some constant k)에 대해서 $P_{GT}$를 적용하고, 높은 빈도수에서는 MLE에 의한 추정을 적용 

* **Solution 2**: $(r, Nr)$에 대해 smoothing function *S*를적용하여 $E(N_r)$에 적용

두 가지 방법을 적용하여, 다음의 개선된 estimator를 정의합니다.

#### 2.3.4 Revised Good-Turing Estimator

If $ C(w_1 \cdots w_n) = r \gt 0 $,
$$ P_{GT}(w_1 \cdots\ w_n) = \frac{r^{\star}}{N}\ \text{where}\ r^{\star}=\frac{(r+1)S(r+1)}{S(r)}$$

if $C(w_1 \cdots w_2) = 0 $,  
$$P_{GT}(w_1 \cdots w_n) = \frac{1-\sum_{r=1}^{\infty N_r\frac{r^\star}{N}}}{N_0} = \frac{N_1}{N} \cdot \frac{1}{N_0}$$

#### 2.3.5 Re-normalization

Solution 1과 Solution 2 혹은 어떠한 복잡한 방법을 적용하더라도, 확률 분포가 잘 구성되도록 정규화해야합니다.
(i) 정규화하는 방법은 간단히 전체 n-gram에 대해 전체 합이 1이 되도록 정규화하는 방법과 (ii) unseen data에 할당한 확률 분포는 유지하고 나머지에
대해서만 정규화하는 방법이 있습니다. 예를 들어, Good-Turing에 의해 unseen data에 할당된 확률이 1/20이라면, seen data에 할당된 확률의 총합이
19/20이 되도록 정규화합니다.

#### 2.3.5 Bigram Example

**Good-Turing Estimator**와 **Re-normalization**을 bi-gram 모델에 적용하는 예제를 살펴보겠습니다.

1. Vocabulary = {a, b, c}
  
2. Number of possible bi-grams = $3^2$ = 9

3. Observed bi-grams = {ba, ab, ba, aa, ac, cb, bc, ca, ac, ca, ac}
	* ba: 2, ab: 1, aa:1, ac: 3, cb:1, bc: 1, ca: 2
	* $N_0=2, N_1=4, N_2=2, N_3=1, 
	* N = \sum_0^3 N_i = 11 $
 
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
        * $P_{GT}(r=0) = 2/11$
        * $\text{constant value} \times P_{GT}(r=1) = 14/110 $ 
        * $\text{constant value} \times P_{GT}(r=2) = 7/110 $
        * $\text{constant value} \times P_{GT}(r=3) = 21/220 $
    * validataion
        * $\sum_{i=0}^{3} P_{GT}(r=i) \times N_i = 2 \cdot \frac{2}{11} + 4 \cdot \frac{7}{110} + 2 \cdot \frac{7}{110} + 1 \cdot \frac{21}{220} = 1 $

### 2.4. Knerser-Ney smoothing

절대 절하
낮은 차수의 `n-gram`은 높은 차수의 `n-gram`의 빈도가 작거나 0일때만 효과가 있다는 것이다.

## 3. Implementation

이제 지난 시간 만들어 두었던 모델을 이용해 smoothing을 진행 해보겠습니다. 우선 모델을 로드하겠습니다.

```python
def load_model(path):
    fin = open(path, 'r')
    return {line.split('\t')[0]: int(line.split('\t')[1]) for line in fin.read().split('\n')}

unigram_path = 'unigram.txt'
unigram_model = load_model(unigram_path)

bigram_path = 'bigram.txt'
bigram_model = load_model(bigram_path)

trigram_path = 'trigram.txt'
trigram_model = load_model(trigram_path)

print(list(unigram_model.items())[:7])
print(list(bigram_model.items())[:6])
print(list(trigram_model.items())[:5])
```

    [('<s>', 94805), ('Emma', 865), ('by', 7601), ('Jane', 302), ('Austen', 2), ('_NUM_', 26919), ('</s>', 94805)]
    [('<s> Emma', 223), ('Emma by', 2), ('by Jane', 3), ('Jane Austen', 2), ('Austen _NUM_', 2), ('_NUM_ </s>', 294)]
    [('<s> Emma by', 1), ('Emma by Jane', 1), ('by Jane Austen', 2), ('Jane Austen _NUM_', 2), ('Austen _NUM_ </s>', 2)]

smoothing을 진행하기 앞서 우리가 생성한 모델이 어떻게 구성되었는지 알아보겠습니다.


```python
from functools import reduce 

unigram_N = len(unigram_model)
bigram_N = len(bigram_model)
trigram_N = len(trigram_model)

print('vocabulary size:', unigram_N)

print('%10s\t%10s\t%15s\t%15s' 
      %('', 'unigram', 'bigram', 'trigram'))
print('%10s\t%10d\t%15d\t%15d'
      %('possible', unigram_N, unigram_N**2, unigram_N**3))
print('%10s\t%10d\t%15d\t%15d' 
      %('observed', unigram_N, bigram_N, trigram_N))
print('%10s\t%10d\t%15d\t%15d' 
      %('unseen',
        unigram_N - unigram_N,
        unigram_N**2 - bigram_N,
        unigram_N**3 - trigram_N))
print('%10s\t%10d\t%15d\t%15d' 
      %('N', 
        sum(unigram_model.values()),
        sum(bigram_model.values()),
        sum(trigram_model.values())))
```

    vocabulary size: 33337
              	   unigram	         bigram	        trigram
      possible	     33337	     1111355569	 37049260603753
      observed	     33337	         574788	        1361050
        unseen	         0	     1110780781	 37049259242703
             N	   2334432	        2239627	        2144822

`gutenberg corpus`로 구성한 `n-gram` 모델은 33337개의 단어로 구성되었고, 각 모델에서 가능한 `n-gram`의 수는 33337^n 개가 가능하겠으며, 
각 모델에서 발견되지 않은 `n-gram`의 확률이 0이 아닌 작은 확률을 갖도록 조정해야겠습니다. `unseen` 데이터는 `observed` 데이터에 비해 상대적으로 
비율이 상당히 큰 것을 알 수 있는데, 이로 인해 `unseen data`에 속하는 각각의 `n-gram`에 아주 작은 확률을 부여한다 할지라도 전체적으로는 많은 
확률을 차지하게 될 가능성이 있게 되는 것입니다.

```python
def get_lap_estimate(N, B, r):
    return (r + 1)/(N + B)

def get_lid_estimate(N, B, r, l):
    return (r + l)/(N + B*l)

def get_gt_estimate(N, r, N_r, model):
    if r == 0:
        N_1 = sum(1 for key, value in model.items() if value == 1)
        return N_1 / N_r / N
    else:
        N_next = sum(1 for key, value in model.items() if value == r+1)
        return (r+1) * N_next / N_r / N
        
def get_expected_frequence(N, P):
    return N * P
    
def smoothing(vocab_size, order, model):
    print('r = f_MLE\t%8s\t%8s\t%8s\t%10s' % ('f_lap', 'f_lid', 'f_gt', 'N_r'))
    for r in range(0, 21):
        print('r =%6d' % r, end='\t')
        
        N = sum(model.values())
        N_r = 0
        if r == 0:
            N_r = vocab_size**order - len(model)
        else:
            N_r = sum(1 for key, value in model.items() if value == r)
            
        #laplace
        p_lap = get_lap_estimate(N, vocab_size**2,r)
        f_lap = get_expected_frequence(N, p_lap)
        print('%.6f' % f_lap, end='\t')
                  
        #lid
        p_lid = get_lid_estimate(N, vocab_size**2,r, 1/2)
        f_lid = get_expected_frequence(N, p_lid)
        print('%.6f' % f_lid, end='\t')

        #good-turing
        p_gt = get_gt_estimate(N, r, N_r, model)
        f_gt = get_expected_frequence(N, p_gt)
        print('%.6f' % f_gt, end='\t')
    
        #N_r
        print('%10d' % N_r)
        
smoothing(len(unigram_model), 2, bigram_model)

```
	r = f_MLE	   f_lap	   f_lid	    f_gt	       N_r
	r =     0	0.002011	0.002007	0.000358	1110780781
	r =     1	0.004022	0.006021	0.368859	    397420
	r =     2	0.006034	0.010036	1.233137	     73296
	r =     3	0.008045	0.014050	2.214153	     30128
	r =     4	0.010056	0.018064	3.161540	     16677
	r =     5	0.012067	0.022078	4.162731	     10545
	r =     6	0.014078	0.026093	5.057682	      7316
	r =     7	0.016089	0.030107	6.053727	      5286
	r =     8	0.018101	0.034121	7.083000	      4000
	r =     9	0.020112	0.038135	8.630877	      3148
	r =    10	0.022123	0.042150	8.866397	      2717

 `Laplace’s law`와 `Lidstone’s law`는 실제 n-gram과의 차이가 심한 반면, `good-turing`에 의한 방법은 그 차이가 훨씬 덜하다는 것을 확인할 수 
있습니다. 

**Note**: 위 구현에서 `re-normalizing`과정이 생략되었습니다. `good turing` 방법을 실제 적용할 때는 상위 n개에 대해서 `smoothing`을 적용하고, 
나머지는 `MLE`로 사용하되, 정규화하는 과정을 추가하기를 권장합니다.  
{: .notice--info}


이번 시간에는 `n-gram model`에서 발생하는 `data sparseness` 문제를 해결하는 방법들에 대해 알아봤습니다.
더 좋은 모델로는 `Kneser–Ney smoothing` , `hierarchical Bayesian` 과 같은 방법들을 찾아보길 권해드립니다.  

[previous post]:https://sept1022.github.io/snlp/2018-10-18-language-model-part-2/
[MLE post]:http://sep1022.github.io/snlp/

