---
layout: post
title:  "pre-training model in NLP"
excerpt: "from word embeddings to pre-training model based on large-scale data set to better capture and represent the contextual information"
date:   2021-01-27 17:00:00
mathjax: true
---



### 1.Word2Vec

Word2Vec is a method to learn the vector representation of word embedding from large datasets. 

It uses the **Skip-gram Model** to predict words within a certain range before and after the current word in the same sentence
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
\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} log p(w_{t+j}|w_t)
\end{equation}
$$

where \\(c\\) is the size of the training context.

$$
\begin{equation}
p(w_{t+j}|w_t) = p(w_{O}|w_I)  = frac{e^{v_{w_O}^{'} v_{w_I}}}{\sum_{w=1}{W} e^{ v_w^{'} v_{w_I} }}
\end{equation}
$$

where \\( v_w\\) and \\( v_w^{'}\\) are the input and output vector representation of \\( w \\), and \\( W\\) is the number of words in the vocabulary. 
the noticeable problem for this is **huge computation cost proportional to the number of 
the word vocabulary for the Softmax function** . To cope with this disadvantage, **Noise Contrastive Estimation (NCE)** is proposed.





### 2.ELMO



### 3.GPT



### 4.BERT



### 5.ERNIE





### 6.Reference

[Word2vec](https://www.tensorflow.org/tutorials/text/word2vec)


