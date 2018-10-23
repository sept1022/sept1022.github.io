---
title: "[NLP] Language Model using n-gram Tutorial - Part 2"
categories:
  - snlp
tags:
  - ngram
  - nlp
  - statistical
  - language model
  - maximum likelihood estimation
sitemap: true
toc: true
toc_label: "Table of Contents"
toc_sticky: true

---

> 본 포스트 시리즈는 FSNLP(Foundation of Statistical Natural Language Processing)의 내용 중에서,
language modeling, HMM(hidden markov, model)을 이용한 POS(part-of-speech) Tagger 구현에 관한 
내용을 정리하는 것을 목적으로 합니다.

[지난 포스트][previous post]에서는 *Language Modeling* 에 대한 기본적인 사항과 n-gram을 만드는 법에 대해서 알아봤습니다.

이번 시간에는 지난 시간에 만들어 두었던 n-gram 모델을 사용하여 MLE(Maximum Likelihood Estimation)에 의한 
확률을 도출하는 법을 알아보겠습니다.

Language Modeling에서 우리가 관심을 갖는 것은 주어진 단어열 $P(w_1 \cdots w_{n-1})$ 의 다음에 출현할
 단어 $P(w_1 \cdots w_{n})$ 의 확률입니다.

$$
P(w_n|w_1 \cdots w_{n-1}) = \frac{P(w_1 \cdots w_{n})}{P(w_1 \cdots w_{n-1})}
$$

이 확률을 구하는 방법에 대해 알아보겠습니다.

### MLE(Maximum Likelihood Estimation, 최대 우도 측정)

**wikipedia:** 
In statistics, maximum likelihood estimation (MLE) is a method of estimating the parameters of
 a statistical model, given observations. MLE attempts to find the parameter values that maximize
 the likelihood function, given the observations.
 The resulting estimate is called a maximum likelihood estimate, which is also abbreviated as MLE.
{: .notice--info}

MLE에 대한 자세한 설명은 [이전 포스팅]({% post_url 2018-10-18-probability-concept %}) 를 참고하세요
{: .notice}

#### MLE estimates from relative frequencies
n-gram의 MLE는 _relative frequency_ 로 구할 수 있습니다. trigram 모델은 선행하는 두 단어 $ w_{n-2}, w_{n-1} $ 를 
$ w_1 $ 을 예측하기 위한 _history_ 혹은 _context_ 로 사용합니다. 만일 특정 코퍼스에서, `come across`의 이후에 출현하는 단어로
`as` 가 8번, `more, a`가 한 번씩 출현했다면, _relative_frequency_ 는 다음과 같이 정의됩니다.

$$
\begin{align*}
P(as) = 0.8 \\
P(a) = 0.1 \\
P(more) = 0.1 \\
P(x) = 0.0
\end{align*}
$$

이렇게 상대적인 빈도를 MLE로 정의할 수 있는데, 아래의 수식에 따라, $$ N $$은 상쇄하고 n-gram의 빈도수만으로 계산할 수 있습니다.

$ P_{MLE}(w_1 \cdots w_n) = \frac{C(w_1 \cdots w_n)}{N} $
$ P_{MLE}(w_n \mid w_1 \cdots w_n) = \frac{C(w_1 \cdots w_n)/N}{C(w_1 \cdots w_n-1)/N} = \frac{C(w_1 \cdots w_n)}{C(w_1 \cdots w_{n-1})} $

### Sentence probability

MLE가 찾는 $ P(x \mid \theta) $ 정의를 다시 보겠습니다. 문장의 단위로 생각하면, $ \theta $ 는 n-gram model을 구성하는 파라미터로, 
각 n-gram의 출현 빈도를 의미하고, $ x $ 는 문장을 구성하는 n-gram 집합을 의미합니다. 지난 포스트에서 논의한 것처럼, $ P(x \mid \theta) $ 는 다음과 같이 정의됩니다.

$$ P(sentence) = \prod_{i}^N P(x_i \mid \theta) $$ 

여기에서, n-gram에 대한 수식으로 치환하면, 

$$ P(sentence) = \prod_{i}^N P(w_i \mid w_{i-n+1} \cdots w_{i-1}) $$

반복된 확률의 곱은 underflow를 야기할 가능성이 있으며, adding 연산이 multiply 연산보다 빠르다는 점으로 인해 log를 취하여 다음의 식으로 사용하기도 합니다.  

$$ \ln P(sentence) = \sum_{i}^N P(w_i \mid w_{i-n+1} \cdots w_{i-1}) $$

### Apply Maximum Likelihood Estimation

n-gram과 sentence에 대해 MLE를 계산할 방법이 마련되었으니, 지난 시간에 구축했던 n-gram model을 활용해보겠습니다.
구축했던 데이터는 `n-gram`과 `n-gram`이 출현한 빈도를 `\t`문자로 구분했습니다. 이 점을 고려하여 `dictionary`로 로딩하면 되겠습니다. 

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
```

잘 로딩되었는지 확인해보겠습니다

```python
print(list(unigram_model.items())[:7])
print(list(bigram_model.items())[:6])
print(list(trigram_model.items())[:5])
```

	[('<s>', 94805), ('Emma', 865), ('by', 7601), ('Jane', 302), ('Austen', 2), ('_NUM_', 26919), ('</s>', 94805)]
	[('<s> Emma', 223), ('Emma by', 2), ('by Jane', 3), ('Jane Austen', 2), ('Austen _NUM_', 2), ('_NUM_ </s>', 294)]
	[('<s> Emma by', 1), ('Emma by Jane', 1), ('by Jane Austen', 2), ('Jane Austen _NUM_', 2), ('Austen _NUM_ </s>', 2)]


이제 n-gram model에 의한 확률을 구할 준비가 되었습니다.
책에서 사용한 `In person she was inferior to both sisters`의 문장에 대해 모델별 확률값을 구해보겠습니다.
	
```python
N = 0
for _, count in unigram_model.items():
    N += int(count)
    
print('N =', N)

def get_n_gram_prob(sentence):
    sentence = sentence.split()
    sentence.insert(0, '<s>')
    sentence.append('</s>')
    
    print('%10s%8s%8s%8s' %('word', 'unigram', 'bigram', 'trigram'))
    for i in range(0, len(sentence)):
    	#unigram
    	if sentence[i] not in unigram_model:
    		print('%5s' % 'unseen')
    	else
        	print('%10s %0.5f' % (sentence[i], unigram_model[sentence[i]]/N), end=' ')
        
        #bigram
        if i > 0:
            hist = sentence[i-1]
            bigram = ' '.join(sentence[i-1:i+1])
            if bigram not in bigram_model:
                print('%0.5f' % 0, end=' ')
            else:
                print('%0.5f' % (bigram_model[bigram]/unigram_model[hist]), end=' ')
        
        #trigram
        if i > 1:
            hist = ' '.join(sentence[i-2:i])
            trigram = ' '.join(sentence[i-2:i+1])
            if hist not in bigram_model:
                print('%7s' % 'unseen', end=' ')
            elif trigram not in trigram_model:
                print('%0.5f' % 0, end=' ')
            else:
                print('%0.5f' % (trigram_model[trigram]/bigram_model[hist]), end=' ')

        print()
sentence = 'In person she was inferior to both sisters'
get_n_gram_prob(sentence)
```

     N = 2334432
          word unigram  bigram trigram
           <s> 0.04061 
            In 0.00066 0.00746 
        person 0.00016 0.00000 0.00000 
           she 0.00283 0.01053  unseen 
           was 0.00738 0.10506 0.25000 
      inferior 0.00002 0.00000 0.00000 
            to 0.01871 0.21053  unseen 
          both 0.00041 0.00027 0.00000 
       sisters 0.00006 0.00000 0.00000 
          </s> 0.04061 0.11194  unseen

### Examine Probability

위 결과 중에서 `0.00000`으로 출력된 결과는 `history`, 즉 (n-1)-gram이 존재하지 않는 경우이고, `unseen`으로 출력된 것은 n-gram이 존재하지 
않는 경우입니다. 우리는 그 동안 training data에 대해 unigram, bigram, trigram에 대한 모델을 생성했고, 위의 문장에 존재하는 n-gram에 대해 
MLE를 적용하여 확률을 계산했습니다.  
 
결과를 살펴보면, _unigram_ 은 _history_ 를 고려하지 않으므로 context에 대한 정보가 전혀 없고, 단지 전체 단어 수에 대한 비율을 반영하고 있습니다. 
반면, _bigram_ 은 현재 단어에 대한 확률을 계산하기 위해 이전의 단어에 대한 정보를 활용하므로, 문맥의 정보를 활용한다는 점에서 unigram model 보다 
나은 모델이라고 할 수 있습니다. 그러나 $ P(\text{person} \mid \text{In}) $ 과 $ P(\text{sisters} \mid \text{both}) $ 의
정보가 존재하지 않아 문장의 확률은 0 이됩니다.

trigram에서는 $ P(\text{was} \mid \text{person, she}) $ 의 경우 unigram, bigram 보다 높은 확률을 부여하는 점에서 꽤 좋다고 할 수 있겠으나, 
대부분의 경우 (n-1)-gram과 n-gram이 존재하지 않아 전제 문장의 확률 측면에서는 도움이 되지 않을 뿐만 아니라,
n-gram에 대한 정보가 거의 없을 때 그 정보를 신뢰하기 어렵다는 문제가 있습니다.

이와 같이 n-gram에 대한 정보가 충분하지 않아 제대로 된 확률을 계산하기 어려운 것을 **Data Sparseness** 문제라고 합니다.
training에 사용한 데이터가 적은 것도 문제겠으나, 데이터를 늘린다할 지라도 언어의 특성상 상위 몇 단어의 빈도가 대부분을 차지하고, 
중요한 단어(Content words)들은 그 빈도가 매우 작다는 점으로 인해 문제 해결이 어렵습니다.   

뿐만 아니라 MLE가 취하는 방식이 관측된 데이터에 대한 가능성을 최대로 하는 것이기 때문에, 발견되지 않은 데이터에 대해 확률을 부여할 수 없다는 점에서,
NLP에서의 MLE에 의한 확률적인 추론을 수행하기에는 문제가 있다는 것을 알 수 있습니다.

이번 포스트에서는 n-gram 모델을 활용하여 최대우도측정 과정을 알아봤습니다. 다음 시간에는 본 포스트에서 언급한 **Data Sparseness** 문제를 극복하기 위한 방법을 알아보겠습니다.

[previous post]:https://sept1022.github.io/snlp/language-model-part-1/