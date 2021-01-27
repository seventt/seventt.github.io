---
layout: post
title:  "pre-training model in NLP"
excerpt: "from word embeddings to pre-training model based on large-scale data set to better capture and represent the contextual information including Word2Vec, ELMO, GPT, BERT and ERNIE"
date:   2021-01-27 17:00:00
mathjax: true
---

The technique of word embedding method and pre-training model is to get the vector representation of word or sentence by exploiting the contextual information of word or sentence as accurately as possible. These numerical value of
vector representation can be subsequently used for the downstream NLP task.

### 1.Word2Vec

Word2Vec is a method to learn the vector representation of word embedding from large datasets. 

It uses the **Continuous Skip-gram Model** to predict words within a certain range before and after the current word in the same sentence.
**Continuous Bag-of-Words Model (CBOW)** is another mothod to train the model by predicting the middle word based on surrounding context words. The context consists of a few words before and after the current (middle) word.

<div class="imgcap">
<img src="/assets/bert/word2vec.png">
<div class="thecap">Skip-gram model and CBOW model.</div>
</div>

Here is the explanation of how to form the training examples using word2vec.

<div class="imgcap">
<img src="/assets/bert/word2vec-generate-sample.png">
<div class="thecap">Process of generating training examples.</div>
</div>

The training objective of the Skip-gram model is to find word representations that are useful for
predicting the surrounding words in a sentence or a document. More formally, given a sequence of
training words \\( w_1,w_2,\cdot,w_T\\), the objective of the Skip-gram model is to maximize the average
log probability

$$
\begin{equation}
\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j}|w_t)
\end{equation}
$$

where \\(c\\) is the size of the training context.

$$
\begin{equation}
p(w_{t+j}|w_t) = p(w_{O}|w_I)  = \frac{e^{v_{w_O}^{\prime} v_{w_I}}}{\sum_{w=1}^{W} e^{ v_w^{\prime} v_{w_I} }}
\end{equation}
$$

where \\( v_w\\) and \\( v_w^{\prime}\\) are the input and output vector representation of \\( w \\), and \\( W\\) is the number of words in the vocabulary. 
the noticeable problem for this is huge computation cost proportional to the number of 
the word vocabulary for the Softmax function . To cope with this disadvantage, **Noise Contrastive Estimation (NCE)** is proposed, which uses **Negative Sampling** to sample 
some negative samples for the input word according to one noise distribution instead of using all the words in the vocabulary to reduce the computation cost.

After training, the vector representation of the words with similar semantic meaning will be more close, in contrast, the vector representation of the words with different semantic meaning will be less similar.
- One problem is: there is one **fixed vector representation** for one word, so it is not suitable for **synonym word (polysemy)**,
- another is: Word2Vec takes less contextual information of current word into consideration, the representation is a little **shallow**,

<div class="imgcap">
<img src="/assets/bert/word2vec-ret.png">
<div class="thecap">The illustration of word embedding for some kind of words .</div>
</div>

### 2.ELMo

**ELMo (Embeddings from Language Models)** is a deep contextualized word representation that models both complex characteristics of word use (e.g., syntax and semantics), a
nd how these uses vary across linguistic contexts (i.e., to model polysemy). These word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), 
which is pre-trained on a large text corpus. They can be easily added to existing models and significantly improve the state of the art across a broad range of challenging NLP problems, including question answering, textual entailment and sentiment analysis.

ELMo representations are;

- **Deep**: The word representations combine all layers of a deep pre-trained neural network.

- **Contextual**: The representation for each word depends on the entire context in which it is used. Instead of using a fixed embedding for each word, ELMo looks at the entire sentence before assigning each word in it an embedding

<div class="imgcap">
<img src="/assets/bert/elmo.gif">
<div class="thecap">The illustration of ELMo.</div>
</div>

### 3.GPT



### 4.BERT



### 5.ERNIE





### 6.Reference

[Word2vec](https://www.tensorflow.org/tutorials/text/word2vec)

[ELMo](https://allennlp.org/elmo)


