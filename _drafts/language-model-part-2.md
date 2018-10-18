---
title: "[NLP] Language Model: ngram over sparse data - Part 2"
categories:
  - snlp
tags:
  - ngram
  - nlp
  - statistical
  - language model
sitemap: false 
---

> 본 포스트 시리즈는 FSNLP(Foundation of Statistical Natural Language Processing)의 내용 중에서,
language modeling, HMM(hidden markov, model)을 이용한 POS(part-of-speech) Tagger 구현에 관한 
내용을 정리하는 것을 목적으로 합니다.

[지난 포스트][previous post]에서는 *Language Modeling* 에 대한 기본적인 사항과 n-gram을 만드는 법에 대해서 알아봤습니다.

이번 시간에는 지난 시간에 만들어 두었던 n-gram 모델을 사용하여 MLE(Maximum Likelihood Estimation)에 의한 
확률을 도출하는 법을 알아보고, Smoothing 기법을 적용하여 보다 나은 통계적 추정(Statistical Estimation)을 
할 수 있는 방법에 대해 알아보겠습니다.

Language Modeling에서 우리가 관심을 갖는 것은 주어진 단어열 $P(w_1 \cdots w_{n-1})$ 의 다음에 출현할
 단어 $P(w_1 \cdots w_{n})$ 의 확률입니다.

$$
P(w_n|w_1 \cdots w_{n-1}) = \frac{P(w_1 \cdots w_{n})}{P(w_1 \cdots w_{n-1})}
$$

이 확률을 구하는 방법에 대해 알아보겠습니다.

### MLE(Maximum Likelihood Estimation, 최대 우도 측정)

**wikipedea:** 
In statistics, maximum likelihood estimation (MLE) is a method of estimating the parameters of
 a statistical model, given observations. MLE attempts to find the parameter values that maximize
 the likelihood function, given the observations.
 The resulting estimate is called a maximum likelihood estimate, which is also abbreviated as MLE.
{: .notice--info}

MLE에 대한 자세한 설명은 [이전 포스팅]({% post_url 2018-10-18-probability-concept %}) 를 참고하세요
{: .notice}

#### MLE estimates from relative frequencies
n-gram의 MLE는 _relative frequency_ 로 구할 수 있습니다. trigram 모델은 선행하는 두 단어 $ w_{n-2}, w_{n-1} $ 를 
$ w_1 $ 을 예측하기 위한 _history_ 혹은 _context_ 로 사용합니다. 만일 특정 코퍼스에서, `come across`의 이후에 발생하는 단어를 보니,
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

이제 n-gram에 대해 MLE를 계산할 수 있으니, 지난 시간에 구축했던 n-gram model을 활용해보겠습니다.

### Examine n-gram Probability
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

likelihood function
If one fixes the observed data, and then considers the space of all pos- sible parameter assignments 
within a certain distribution (here a trigram model) given the data, then statisticians refer to this as a likelihood function.
The maximum likelihood estimate is so called because it is the choice of parameter values which gives the highest probability 
to the training corpus.4 The estimate that does that is the one shown above. It does not waste any probability mass on events 
that are not in the train- ing corpus, but rather it makes the probability of observed events as high as it can subject to 
the normal stochastic constraints. But the MLE is in general unsuitable for statistical inference in NLP. 
The problem is the sparseness of our data (even if we are using a large corpus). While a few words are common, 
the vast majority of words are very uncommon – and longer n-grams involving them are thus much rarer again. 
The MLE assigns a zero probability to unseen events, and since the probability of a long string is generally 
computed by multiplying the probabilities of subparts, these zeroes will propagate and give us bad (zero probability) 
estimates for the probability of sentences when we just happened not to see certain n-grams in the training text.5 
With respect to the example above, the MLE is not capturing the fact that there are other words which can follow comes across, for example the and some.

MLE는 관측된 데이터가 발생할 가능성을 최대로 하는 것이기 때문에 , 우리가 다루는 NLP에서의 확률적인 추론에 부적합하다.

[previous post]:https://sept1022.github.io/snlp/language-model-part-1/