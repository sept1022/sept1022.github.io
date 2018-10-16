---
title: "[NLP] Language Model: gram over sparse data - Part 1"
categories:
  - snlp
tags:
  - ngram
  - nlp
  - statistical
  - language model 
---

> 본 포스트 시리즈는 FSNLP(Foundation of Statistical Natural Language Processing)의 내용 중에서,
language modeling, HMM(hidden markov, model)을 이용한 POS(part-of-speech) Tagger 구현에 관한 
내용을 정리하는 것을 목적으로 합니다.


**Statistical NLP**은 자연어 처리 영역에서 확률적인 추론(inference)을 하는 것을 목적으로 합니다.
일반적으로 *Statistical inference*는 데이터를 통해 확률 분포를 학습(training)하고, 학습된 분포에 따라 
추론(inference)을 수행하는 것으로 구성됩니다.

<div class="mermaid">
graph LR
        A["Training Data"]-->B["Model"]
        B["Model"]-->C["Inference for a new instance"]
</div>

대부분의 통계적인 방법론은 과거의 축적된 데이터로부터 모델을 생성하고, 새로운 데이터에 대해 학습된 모델에 기반하여
추론을 수행하게 됩니다. 그 중에서, 우리가 살펴볼 문제는 `Language Modeling`입니다. 
`Language Modeling`은 연속된 단어의 뒤에 어떤 단어가 올것인지를 예측하는 문제로, 음성인식(speech recognition),
문자 인식(OCR, optical character recognition), 문자 교정(spelling correction),
통계 기계 번역(statistical machine translation)과 같은 언어처리를 수행하기 위한 가장 기본적인 과정입니다. 

우리가 살펴볼 `Language Modeling`은 다음의 세가지 영역으로 구성됩니다.
- **n-gram modeling**: dividing the training data into equivalence classes
- **Smoothing**: finding a good statistical estimator for each equivalence class 
- **Interpolation**: combining multiple estimators   

## n-gram models

주어진 단어열의 다음에 출현할 단어의 확률을 예측하는 것은 다음의 식을 측정하는 것으로 정의될 수 있습니다.

$$
P(w_n|w_1,\cdots, w_{n-1})
$$ 

즉, *history*($$w_1,\cdots, w_{n-1}$$)에 분류를 적용하여 다음에 출현할 $$w_n$$를 예측하는 것이죠.  
이 예측을 수행하기 위한 *history*에 대한 학습 결과를 n-gram model이라고 이해하면 좋겠습니다.

*n-gram* 에서의 n은 고려할 단어의 수입니다. 만일 history를 마지막 n-1개로 모델을 구성했다면, 
이것을 $$(n-1)$$ oder Markov model 혹은 n-gram word model 이라고 합니다. 
n-gram 에서의 마지막 단어는 예측할 단어입니다. 보통 $$n\ =\ 2,3,4$$로 설정하는데, 
각각 *bigram, trigram, four-gram*이라고 불립니다.

예를 들어, 다음의 문장에서 green의 다음에 출현할 단어를 예측한다고 하면, 

> Sue swallowed the large green _

|Model|Equation|
|-----|--------|
|bigram|$$ P(w_n\|green)$$|
|trigram|$$ P(w_n\|large,green)$$|
|four-gram|$$ P(w_n\|the,large,green)$$|

즉 _Vocabulary_ 에 존재하는 단어 중에서 모델의 확률을 최대로 하는 $$w_x$$를 찾아야 하는 것입니다.

> $$argmax_{x \in \mathcal{V}}P(w_x|history)$$


그렇다면 history에 해당하는 n-1은 얼마나 고려해야 할까요?
단어의 수가 20,000개일 때의, 각 모델에서 가능한 n-gram의 수를 계산하면 아래의 표와 같습니다.

|Model|Parameters|
|-----|----------|
|1st order (bigram model)| $$20,000 \times 19,999 = 400\ \text{million}$$|
|2nd order (trigram model)| $$20,000^2 \times 19,999 = 8\ \text{trillion}$$|
|3st order (four-gram model)| $$20,000^3 \times 19,999 = 1.6 \times 10^{17}$$|

four-gram을 비롯하여 더 높은 차수의 n-gram을 학습하기 위해서는 어마어마한 양의 학습 데이터가 필요하다는 것을 알 수 있습니다.
바꾸어 말하면, 데이터가 충분하지 않으면 n-gram 또한 충분히 학습되지 않고, 예측을 잘 수행할 수 없게 됩니다. 이런 이유로, bigram과 trigram이 주로 사용됩니다.

물론 Parameter의 수를 줄이기 위한 방법들이 존재합니다. 단어의 변화가 발생하는 뒷부분을 제거(stemming)하거나,
thesaurus나 clustering 정보를 활용하여 의미적으로 비슷한 단어를 하나의 단어로 취급하여 n-gram의 수를 줄일 수는 있습니다만, 
본 포스트가 다루는 범위가 넘어가므로, 다루지 않겠습니다.

## Examine Corpus

nltk에서 제공하는 gutenburg project의 코퍼스를 대상으로 n-gram model을 만들어보겠습니다.
우선 데이터가 어떻게 생겼는지 살펴볼까요?

```python
from nltk.corpus import gutenberg
from collections import Counter
```

책에서 다룬 것처럼 `persuasion`은 test_set으로 따로 두겠습니다.

```python
training_set = gutenberg.fileids().copy()
test_set = 'austen-persuasion.txt'
training_set.remove(test_set)
print('training_set:', training_set)
print('test_set:', test_set)
```

    training_set: ['austen-emma.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']
    test_set: austen-persuasion.txt

```python
sents = gutenberg.sents(training_set)
words = gutenberg.words(training_set)
print('sents:', len(sents), 'words:', len(words), 'avg length:', len(words)/len(sents))
```

    sents: 94805 words: 2523442 avg length: 26.617182638046515

총 94,805 문장에, 2,523,443개의 단어가 존재하고, 각 문장은 약 26개의 단어로 구성되어 있습니다.

unigram에 대한 분포를 볼까요?

```python
unigram = Counter(words)
print('words:', len(words), 'vocabulary:', len(unigram))
```

    words: 2523442 vocabulary: 50738

2,523,443의 단어는 50,738개 유니크한 단어로 이루어져 있습니다. 
단어들의 빈도를 볼까요?

```python
unigram.most_common()[:20]
```

    [(',', 179341),
     ('the', 122628),
     ('and', 76107),
     ('.', 71005),
     ('of', 67514),
     (':', 47282),
     ('to', 43668),
     ('a', 30975),
     ('in', 30613),
     ('I', 29097),
     ('that', 26436),
     (';', 26039),
     ('he', 21462),
     ('his', 19960),
     ("'", 19348),
     ('it', 18877),
     ('was', 17228),
     ('And', 16471),
     ('with', 16184),
     ('for', 16165)]
     
데이터를 보니, 최고빈도를 갖는 단어는 전치사, 관사, 접속사라는 것을 확인할 수 있습니다.

## Pre-processing

Language Modeling을 다룰 때 가장 중요한 것은 data sparseness를 어떻게 극복할 것인가 하는 문제입니다.

nltk에서 제공하는 `gutenberg.sents()`는 이미 기호가 제거된 형태로 잘 분리된 형태지만, 
기호는 더 많은 n-gram의 수를 요구하므로, string.punctuation에 해당하는 기호는 제거하도록 하겠습니다.
그리고, 문장의 시작과 끝을 표시하는 `<s>`s와 `</s>`를 추가하겠습니다.

실제 적용하는 과정에서는, 보통 모든 단어에 대한 n-gram을 생성하지 않고, 빈도수가 일정 이상되는 단어에 대해서
n-gram을 생성하고, 나머지 단어는 `UNK` 태그로 Out-Of-Vocabulary로 취급합니다. 또한, 어떤 학습 데이터도 모든 
숫자를 포함할 수 없으므로, 숫자는 `NUM`으로 치환하여 사용하도록 하겠습니다. 

```python
import string
punct = string.punctuation
print('punctuation:', punct)

unk_list = [w for w, c in unigram.items() if c == 1]
print('uncommon word:', unk_list[:10])

unk_list = set(unk_list)

def preprocessing(sents):
    for sent in sents:
        for i, word in enumerate(sent):
            if word in punct:
                sent.remove(word)
            elif word.isdigit():
                sent[i] = '_NUM_'
            elif word in unk_list:
                sent[i] = '_UNK_'
        sent.insert(0, '<s>')
        sent.append('</s>')
        
sample_sents = list(sents[:5]).copy()
preprocessing(sample_sents)
print(sample_sents)

```
	punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
	uncommon word: ['1816', 'valetudinarian', 'Matrimony', 'chatted', 'curtseys', 'bangs', 'Dirty', 'drizzle', 'Mitchell', 'Success']

	[['<s>', 'Emma', 'by', 'Jane', 'Austen', '_NUM_', '</s>'] 
	['<s>', 'VOLUME', 'I', '</s>'], 
	['<s>', 'CHAPTER', 'I', '</s>'], 
	['<s>', 'Emma', 'Woodhouse', 'handsome', 'clever', 'and', 'rich', 'with', 'a', 'comfortable', 'home', 'and', 'happy', 'disposition', 'seemed', 'to', 'unite', 'some', 'of', 'the', 'best', 'blessings', 'of', 'existence', 'and', 'had', 'lived', 'nearly', 'twenty', 'one', 'years', 'in', 'the', 'world', 'with', 'very', 'little', 'to', 'distress', 'or', 'vex', 'her', '</s>'],
	['<s>', 'She', 'was', 'the', 'youngest', 'of', 'the', 'two', 'daughters', 'of', 'a', 'most', 'affectionate', 'indulgent', 'father', 'and', 'had', 'in', 'consequence', 'of', 'her', 'sister', 's', 'marriage', 'been', 'mistress', 'of', 'his', 'house', 'from', 'a', 'very', 'early', 'period', '</s>']]

이제 n-gram model을 생성할 데이터가 준비가 되었습니다.

## Building n-gram model

```python
preprocessed_sents = list(sents).copy()
preprocessing(preprocessed_sents)

def make_ngram(sents, n, container):
    for i, sent in enumerate(sents):
        for i in range(0, len(sent)-n+1):
            gram = ' '.join(sent[i:i+n])
            container.setdefault(gram, 0)
            container[gram] += 1
```

```python
unigram = {}
make_ngram(preprocessed_sents, 1, unigram)
print('unigram: ', list(unigram.items())[:10])

bigram = {}
make_ngram(preprocessed_sents, 2, bigram)
print('bigram: ', list(bigram.items())[:10])

trigram = {}
make_ngram(preprocessed_sents, 3, trigram)
print('trigram: ', list(trigram.items())[:10])
```

    unigram: [('<s>', 94805), ('Emma', 865), ('by', 7601), ('Jane', 302), ('Austen', 2), ('_NUM_', 26919), ('</s>', 94805), ('VOLUME', 3), ('I', 29097), ('CHAPTER', 291)]
    bigram: [('<s> Emma', 223), ('Emma by', 2), ('by Jane', 3), ('Jane Austen', 2), ('Austen _NUM_', 2), ('_NUM_ </s>', 294), ('<s> VOLUME', 3), ('VOLUME I', 1), ('I </s>', 117), ('<s> CHAPTER', 276)]
    trigram: [('<s> Emma by', 1), ('Emma by Jane', 1), ('by Jane Austen', 2), ('Jane Austen _NUM_', 2), ('Austen _NUM_ </s>', 2), ('<s> VOLUME I', 1), ('VOLUME I </s>', 1), ('<s> CHAPTER I', 8), ('CHAPTER I </s>', 10), ('<s> Emma Woodhouse', 1)]

첫 번째 문장 *Emma by Jane Austen 1816*에 대해 n-gram 모델이 잘 생성된 것을 확인할 수 있습니다.   
이제 생성된 모델을 파일에 저장하겠습니다.

```python
def write_model(path, container):
    with open(path, 'w') as out:
        out.write('\n'.join([gram + '\t' + str(count) for gram, count in container.items()]))

write_model('./unigram.txt', unigram)
write_model('./bigram.txt', bigram)
write_model('./trigram.txt', trigram)
```
 
이번 포스트에서는 language modeling의 전반적인 개념과 n-gram을 만드는 방법을 알아보았습니다.
다음 포스트에서는 좋은 예측기(good statistical estimator)를 찾는 방법에 대해 알아보겠습니다. 