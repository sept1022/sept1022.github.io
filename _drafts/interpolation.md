---
title: "[NLP] Language Model using n-gram Tutorial - Part 4"
categories:
  - snlp
tags:
  - ngram
  - nlp
  - statistical
  - language model
  - interpolation
sitemap: false
toc: true
toc_label: "Table of Contents"
toc_sticky: true
---

## 3 Combining Estimators

위에서 알아본 방법들은 모두 n-gram의 출현빈도를 기반으로 `unseen data`에 확률을 할당하기 위한 방안들이었습니다.
이제부터 알아볼 방법은 한 번도 출현하지 않거나 거의 출현하지 않는 `n-gram`에 대한 확률을 낮은 차수의 `n-gram`의 정보를 활용하여 보다 나은 확률 추정을 하고자 하는 방법들입니다.
직관적으로, (n-1)-gram이 희귀하다면 작은 확률을 n-gram에 적용하고, (n-1)-ngram이 적당히 존재한다면 `n-gram`에 높은 확률을 적용할 수 있을 것입니다.

**Note**: 일반적으로 n-gram 모델링에서, 다양한 차수의 n-gram model을 결합하는 것이 좋은 모델을 생성하는 좋은 방법입니다.
{: .notice--info}

### 3.1 Simple Interpolation

만일 확률값을 계산할 수 있는 다양한 모델이 존재하면, 그 모델들이 기여하는 바에 따라 적당한 가중치를 설정하여 하나의 확률 값을 도출하도록 결합할 수 있습니다.
보통은 이런 방법을 `(finite) mixture model`이라고 하고, NLP 영역에서는 `linear interpolation`이라고 합니다. 또한, 이런 모델을 구성하는 방법으로
가능한 모든 하위 집합을 모두 결합하는 경우에는 `deleted interpolation`이라고 합니다.

예를 들어, trigram 모델을 `deleted interpolation`의 방법으로 가장 기본적인 방법은 아래와 같이 정의됩니다.

$$
\begin{align*}
 P_{li}(w_n \mid w_{n-2}, w_{n-1}) & = \lambda_{1}P_{1}(w_n) + \lambda_{1}P_{1}(w_n \mid w_{n-1}) \\
								   & \ + \lambda_{1}P_{1}(w_n |w_{n-1}, w_{n-2})
\end{align*}
$$

$ \text{where}\ 0 \leq \lambda_{i} \leq 1\ \text{and}\ \sum_i \lambda_i = 1 $

가중치는 중요도에 따라 사람의 손에 의해 설정될 수 있지만,
[Expectation Maximization]([https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm]) 알고리즘에 의해 최적화된
가중치를 자동으로 찾을 수도 있습니다. 이 방법은 따로 포스트를 만들어서 알아보겠습니다.

### 3.2 Katz's backing-off

다른 방법으로, `n-gram`이 존재하지 않을 때 낮은 차수(`lower order`)의 n-gram으로 `후퇴(baking-off)`하는 방법입니다. 예를 들어,
`trigam`의 정보를 찾을 때, 존재하지 않으면 반복적으로 `bigram`, `unigram`의 순으로 후퇴하여 확률값을 추정하게 됩니다.

$$ P_{bo}(w_i \mid w_{i-n+1} \cdots w_{i-1})= \begin{cases}
												\text{if}\ C(w_{i-n+1} \cdots w_{i-1}) \gt k, \\
												(d_{w_{i-n+1} \cdots w_{i-1}}) \frac{C({w_{i-n+1} \cdots w_{i}})}{C({w_{i-n+1} \cdots w_{i-1}})} \\
												\text{otherwise}, \\
												\alpha_{w_{i-n+1} \cdots w_{i-1}}P_{bo}(w_i \mid w_{i-n+2} \cdots w_{i-1}) \tag{3}
											  \end{cases} $$

위 식에서 $k$는 보통 0이나 1로 설정하는데, 출현하지 않았거나, 1번 출현한 것을 의미합니다. 따라서 빈도수가 k보다 많은 경우는 첫 번째 수식에 의해
MLE가 적용되는데, `unseen data`에 대학 확률 공간을 확보하기 위해서 function $d$에 의해 절하(discounted)된 확률 값을 취하고, 절하된 양을
`unseen data` 할당하게 됩니다. 여러가지 절하방법이 있겠으나, `katz`는 `Good-Turing`에 의한 방법을 사용합니다.

위 수식에서 이해가 필요한 부분은 back-off를 하는 기준인 $k$, 절하를 얼마나 해야하는가를 정하는 $d$, back-off에서 구해진 확률을 어느 비율로 취하는
가를 정하는 $\alpha$를 이애해야 합니다. 이 설명은 [위키]([https://en.wikipedia.org/wiki/Katz%27s_back-off_model])의 내용을 살펴보며 이해를 진행하겠습니다.

#### Back-off Criteria: $k$
back-off를 하기 위한 최소 빈도수를 지정하기 위한 파라미터입니다. 보통은 0이나 1로 설정하는데, 0인 경우는 한 번도 출현하지 않는 경우에만 두 번째 수식을 통해
back-off를 수행하겠다는 의미입니다.

#### The Amount of Discounting: $d$
Good-Turing에 의해 조정된 빈도수 $r^\star$는 \ref{equa_2}에서 정의한 것처럼 $r^\star = \(r+1)\frac{N_{r+1}}{N_r}$로 구하게 됩니다.
따라서 여기에서의 조정된 비율은 $\frac{r^\star}{r} = \frac{C^{\star}(w_{i-n+1} \cdots w_i)}{C(w_{i-n+1} \cdots w_i)} $의 비율인 $d$가 되는 것입니다.
즉, Good-Turing에 의해 조정된 비율만큼 back-off에서 사용하겠다는 것이지요.

#### Back-Off Weight: $\alpha$
Parameter $d$에 의한 값을 어느 곳에 할당해야 할까요? 확률을 할당해야 하는 양 $\beta$는 아래와 같이 정의됩니다.

$$\beta_{w_{i-n+1} \cdots w_{i-1}} = 1 - \sum_{\{w_i:C(w_{i-n+1} \cdots w_i) > k\}} d_{w_{i-n+1} \cdots w_i}  \frac{C({w_{i-n+1} \cdots w_i})}{C(w_{i-n+1} \cdots w_{i-1})}$$

우선 좌변을 살펴보겠습니다. $ w_{i-n+1} \cdots w_{i-1} $ 는 `n-gram` 자체를 의미합니다. 따라서, 현재 `n-gram`이 파라미터 d에 의해 절하된 양을 의미합니다.
우변의 $\sum$의 밑은 $C(w_{i-n+1} \cdots w_i) \gt k$, `n-gram`의 빈도가 `k` 값 초과인 경우에 대하여, `MLE`의 절하된 확률값들을 취하겠다 것입니다.

절하되었던 확률 공간을 할당하기 위한 Back-off weight $\alpha$는 다음과 같이 정의할 수 있습니다.

$$\alpha_{w_{i-n+1} \cdots w_{i -1}} = \frac{\beta_{w_{i-n+1} \cdots w_{i -1}}} {\sum_{ \{ w_i : C(w_{i-n+1} \cdots w_{i}) \leq k \} } P_{bo}(w_i \mid w_{i-n+2} \cdots w_{i-1})}$$
우변의 $\sum$의 밑은 $C(w_{i-n+1} \cdots w_i) \leq k$, `n-gram`의 빈도가 `k` 값 이하인 경우에 대해,
(n-1)-gram의 back-off된 확률 값의 합으로 정규화하는 과정이라는 것을 알 수 있습니다.