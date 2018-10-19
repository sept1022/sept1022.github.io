---
title: "[Statistics] Probability Concept and Maximum Likelihood Estimation"
categories:
  - Statistics
tags:
  - statistics
  - probability
  - Maximum Likelihood Estimation
sitemap: true
toc: true
toc_label: "Table of Contents"
toc_sticky: true
---

> 본포스트는 Maximum Likelihood Estimation을 간결하고 이해하기 쉽게 설명한 [Jonny Brooks Post][jonny brooks post]와 
[Robotics: Estimation and Learning][Coursera Lecture]를 참고하여 작성된 것임을 밝힙니다.

## Probability Definitions and Notation

확률은 어떤 사건(event)에 대한 것이다. 주사위를 던지는 것이나, 주머니에서 특정한 색의 공을 꺼내는 것을 예로 들 수가 있다.
주사위를 던지는 경우 우리는 주사위가 어떤 것이 나올지 알 수 없는데, 이렇게 무작위인 이벤트의 결과를 표현하는 변수를 
*random variable* 이라고 한다. 

우리는 이런 *random variable* 이 특정한 값을 취할 확률을 알고 싶어한다. 예를 들어, 6개의 면을 가진 주사위를 던졌을 때, 
3이 나올 확률은 얼마인가? 여기에서, 각 면이 나올 확률이 같다는 조건은 매우 중요한 정보인데, 이를 통해 위 질문의 답은 1/6이라는 것을
직관적으로 알 수 있다.

위 내용을 수학적으로 표현하기 위해서는, *random variable* 이 주사위를 굴린 결과라는 것을 이해할 필요가 있다.
그리고 *random variable* 은 보통 대문자로 표기하는데, 이 *random variable* 로 $$X$$ 로 사용하면,
주사위를 던졌을 때 3이 나올 확률은 $$X=3$$일 확률을 구하는 것과 같은 표현이고, 더욱 짧게 *Probability* 를 의미하는 $$P$$ 를
사용하여 $$P(X=3)$$ 이라고 표현할 수 있다.

## The 3 types of probability

*Marginal Probability(주변 확률)*
: 사건 A에 대해, Marginal Probability는 A가 발생할 확률 $$P(A)$$ 을 의미한다.
: 예) 포커에서 무작위로 카드를 뽑았을 때, red 카드일 확률: $$P(red) = 0.5$$
 
*Joint Probability(결합 확률)*
: 둘 혹은 그 이상의 사건이 동시에 발생할 확률로, 두 사건이 겹치는 부분을 의미한다. 사건 A와 B에 대해 두 사건의 
결합 확률은 $$P(A \cap B)$$ 로 표현한다.
: 예) 포커에서 무작위로 카드를 뽑았을 때, red이면서 동시에 4일 확률 = 2/52 = 1/26 이다

*Conditional Probability(조건부 확률)*
: 조건에 해당하는 사건이 발생한 상태에서 사건이 발생할 확률이다. 사건 B가 발생한 후, 사건 A가 발생할 확률은 $$P(A \mid B)$$로 표현한다
: 예) 포커에서 뽑은 카드가 red라는 것을 알고 있을 때, 그 중 4를 뽑을 확률은 $$P(4 \mid red)=2/26=1/13$$ 이다. 52장의 카드중에서 
한 장을 뽑았을 때 빨간색일 카드는 모두 26가지의 경우의 수를 갖는다. 26개의 경우의 수 중에서 4를 뽑을 확률을 찾는 것이다. 
 
*Linking the probability types: The general multiplication rule*
: 위의 세 가지 확률은 아래 수식의 관계를 갖는다.
  
$$P(A \mid B)=\frac{P(A \cap B)}{B}$$

### Distinguishing joint probability and conditional probability
> 포커의 예에서 한장을 뽑았을 때 빨간색이고 4일 카드일 결합확률($$P(red,4)$$)을 찾는 경우는 전체 카드 52장에서
무작위로 한장을 뽑는 경우이므로, 52장의 카드 중에서 diamond 4 혹은 hearts 4인 두 장을 뽑는 경우만 존재하므로, 확률은 2/52 = 1/26이 된다.

> 뽑을 카드가 빨간색인 것을 이미 알고 있는 상태에서, 4를 뽑는 경우의 조건부확률($$P(4 \mid red)$$는 다음과 같이 생각해보자.
카드를 뽑기 전에 카드의 종류를 골라낼 수 있어서, 52장의 카드 중 빨간색 카드 26장 골라낸 후, 한 장을 뽑는 것이다.
따라서, 확률은 2/26 = 1/13이 된다.

### Using general multiplication rule

$$P(A \cap B)$$는 위에서 정의한 *general multiplication rule* 에 따라 $$P(A \mid B) \times P(B)$$ 로 구할 수 있다.
사건 A를 4가 뽑이는 경우, 사건 B를 빨간색 카드인 경우로 보면 $$P(A|B)=2/26=1/13$$ 이고 $$P(B) = 1/2$$이므로 
$$P(A \cap B) = 1/13 \times 1/2 = 1/26$$ 이다.

## MLE(Maximum Likelihood Estimation)
Machine Learning에서 실제 관측된 데이터의 겨롸들을 잘 설명하기 위해 Model을 사용한다. 예를 들어, 현재의 구독자가 구독을 취소할 지 여부를
판단하기위해 `Random Forest model`을 사용하거나, 광고에 사용된 예산에 따른 수익 실적의 변화를 예측하기 위해 `linear model`을 사용하는데, 
각 모델은 해당 모델을 정의하기 위한 고유의 파라미터들을 포함하고 있다.

`Linear Model`은 $$y=mx+c$$ 로 정의하는데, 광고에 따른 수익 실적의 예에서 $$x$$ 는 사용된 예산을 나타내고, $$y$$ 는 그에 따른 수익 실적을
나타낸다. 여기에서 $$m$$ 과 $$c$$ 가 이 모델의 파라미터이다. `Linear Model`은 이 파라미터에 의해 같은 입력 $$x$$ 에 대해 다른 결과를 도출한다.

### Intuitive explanation of maximum likelihood estimation
MLE(Maximum likelihood estimation)은 위와 같은 모델에서 파라미터를 찾는 방법인데, 모델이 도출하는 결과들이 실제로 관측된 데이터를 생설할
가능성이 최대가 되도록 파라미터를 찾는 것이다. 본 블로그에서는 Gaussian Model에 대한 MLE를 진행해보겠습니다.

### Gaussian Model

>**wikipedia**: In probability theory, the normal (or Gaussian or Gauss or Laplace–Gauss) distribution is a very common continuous
probability distribution. Normal distributions are important in statistics and are often used in the natural and 
social sciences to represent real-valued random variables whose distributions are not known.
A random variable with a Gaussian distribution is said to be normally distributed and is called a normal deviate.

Gaussian Distribution은 다음과 같이 정의됩니다.

$$ f(x \mid \mu, \sigma^2) = {\frac{1}{\sqrt{2 \pi \sigma^2}}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

위 수식에 대해 알아보기 위해 위의 Probability Density Function을 정의합니다.

```python
def get_gaussian_pdf(x, mu, sigma):
    numerator = np.exp(-1 * (x-mu)**2 / (2 * sigma))
    denomenator = np.sqrt(2 * np.pi * sigma)
    return numerator / denomenator
```

-5부터 5사이의 100개의 데이터를 생산하고, 다양한 $$\mu, \sigma$$ 에 의해 도출되는 데이터를 확인해봅시다.

```python
x = np.linspace(-5, 5, 100)

parameter = [(0, 0.2), (0, 1.0), (0, 5.0), (-2, 0.5)]
for mu, sig in parameter:
    plt.plot(x, get_gaussian_pdf(x, mu, sig))    

plt.legend([ "μ: %.2f var: %.2f" % (mu, sigma) for mu, sigma in parameter])
plt.show()
```

![png](/assets/img/gaussian_distribution.png)

정의에 따라, 그래프는 벨모양을 하고 있고, 평균에 가장 높은 값이 위치하고 좌우로 갈수록 낮은 값이 형성되는 것을 확인할 수 있습니다.  

이제, 정규 분포($$\mu=0, \sigma=1$$)를 따르는 데이터 10,000,000개를 생성하고 이 데이터에 대해 MLE를 진행해 보겠습니다. 

```python
mu, sigma = 0, 1
observed_data = np.random.normal(mu, sigma, 1000000)
```

실제 만들어진 데이터가 원하는 분포를 하고 있는지 보겠습니다.

```python
count, bins, ignored = plt.hist(observed_data, 100, density=True, color='orange')
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='blue')
plt.show()
```

![png](/assets/img/gaussian_distribution_observed.png)

데이터가 준비가 되었으니, MLE를 진행할 준비가 되었습니다. 위에서 제가 mu, sigma를 정의하고 데이터를 생산했지만, 지금부터는 이 값을 모르는 상태라고
가정하고 진행하겠습니다.

가능도(likelihood)는 다음과 같이 정의합니다. 여기서 $$x$$ 는 observed data를, $$\theta$$ 는 모델에 정의된 파라미터 집합을 의미합니다.

$$ \text{Likelihood:}\ \ P(x \mid \theta) $$

Gaussian은 $$\mu, \sigma$$ 를 파라미터로 취하니, 다음과 같이 됩니다.

$$ \text{Likelihood:}\ \ P(x \mid \mu, \sigma) $$

우리가 수행하고자 하는 것은 위의 주어진 데이터에 대해 위의 likelihood를 최대로 하는 파라미터 $$\mu, \sigma$$ 를 찾는 것으로 다음과 같이 정의합니다.

$$ \hat{\mu}, \hat{\sigma} = \operatorname*{arg\,max}_{\mu, \sigma} P(x \mid \mu, \sigma) $$

위 식에서 $$P(x \mid \mu, \sigma)$$ 는 관측된 모든 데이터에 대한 전체 확률을 구하는 것으로, 모든 데이터들의 결합 확률을 의미합니다.
이것은 위 식을 계산하기 어렵게 만들기 때문에, 독립 가정을 설정하여 문제를 단순화합니다. 독립 가정은 각 사건이 독립적으로 발생하는 것으로 간주하는 것인데
위식의 각 데이터들의 주변확률을 곱해주면 됩니다. 

$$ \hat{\mu}, \hat{\sigma} = \operatorname*{arg\,max}_{\mu, \sigma} \prod_{i=1}^{N} P(x_i\mid\mu,\sigma)$$

여기에서 우리가 찾고자 하는 값은 위 수식을 최대화하는 값이기 때문에, 최대화하는 조건을 만족하면서도 계산을 단순화하도록
로그를 취하여 확률의 곱을 합으로 변경합니다. log 함수는 단조 함수이기 때문에 우리가 찾고자 하는 값에 영향이 없을 것입니다.

**_Note_**: 확률은 0 ~ 1사이의 값을 갖기 때문에, 반복된 확률의 곱은 컴퓨터가 표현할 수 있는 수를 넘어가므로 log를 적용하는 방법을 쓰기도 합니다.
{: .notice--info}

$$ 
\begin{align*}
\hat{\mu}, \hat{\sigma} & = \operatorname*{arg\,max}_{\mu, \sigma} \ln\left\{\prod_{i=1}^{N} P(x_i\mid\mu,\sigma)\right\} \\
						   & = \operatorname*{arg\,max}_{\mu, \sigma} \sum_{i=1}^{N}\ln P(x_i\mid\mu,\sigma)
\end{align*}				
$$

가우시안 함수를 수식에 적용하고, log를 각 항에 전개합니다.
    
$$
\begin{align*}
\hat{\mu}, \hat{\sigma} & = \operatorname*{arg\,max}_{\mu, \sigma} \sum_{i=1}^{N} \left \{ \ln {\frac{1}{\sqrt{2 \pi \sigma^2}}}e^{-\frac{(x-\mu)^2}{2\sigma^2}} \right \} \\ 
	 & = \operatorname*{arg\,max}_{\mu, \sigma} \sum_{i=1}^{N} \left \{ {-\frac{(x_i-\mu)^2}{2\sigma^2} - \ln\sigma -\ln\sqrt{2\pi}} \right \}
\end{align*} 
$$
    
log를 전개한 항 중에서 마지막 $$\ln\sqrt{2\pi}$$ 는 항상 같은 값으로 수식을 최대화 하는데 영향이 없으므로 제거합니다. argmax가 취하는 수식을 변경하여
argmin의 형태로 변경하면, 함수의 최소값을 찾는 방법을 따를 수 있습니다.

**_Note_** : 최대값을 찾는 경우라도, 2차 함수가 최대인 기울기가 0인 상태를 찾으면 되는 것이므로 동일한 문제입니다.
{: .notice--info}

$$\hat{\mu}, \hat{\sigma} = \operatorname*{arg\,min}_{\mu, \sigma} \sum_{i=1}^{N} {\frac{(x_i-\mu)^2}{2\sigma^2} + \ln\sigma } $$
    
편미분을 적용하여 최소값에 관한 수식을 유도합니다. 

$$\frac{\delta J(\mu, \sigma)}{\delta \mu} = 0 \to \hat{\mu}, \quad \hat{\mu}=\frac{1}{N}\sum_{i=1}^{N} x_i$$

$$\frac{\delta J(\hat{\mu}, \sigma)}{\delta \sigma} = 0 \to \hat{\sigma}, \quad \hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i-\hat{\mu} )^2$$

**_Note_** : 혹시 위 미분 과정이 필요하다면 메일로 연락주시기 바랍니다. 보충 설명을 추가하겠습니다.
{: .notice--info}

최종적으로 만들어진 $$\hat{\mu}, \hat{\sigma}$$가 실제 데이터를 생성할 때 적용했던 파라미터에 부합하는지 확인해 보겠습니다. 

```python
def parameter_estimation(observed_data):
    mu = reduce(lambda x, y: x + y, observed_data) / len(observed_data)
    sigma = reduce(lambda x, y: x + (y-mu)**2, observed_data) / len(observed_data)
    return mu, sigma

parameter_estimation(observed_data)
```

    (0.0014132501812843044, 0.99799723778905336)

이번 시간은 확률의 기초적인 내용과, Machine Learning에서 등장하는 MLE에 대하여 알아보았습니다.

[jonny brooks post]: https://towardsdatascience.com/probability-concepts-explained-introduction-a7c0316de465
[Coursera Lecture]: https://www.coursera.org/lecture/robotics-learning/1-2-2-maximum-likelihood-estimate-mle-Uf3mO