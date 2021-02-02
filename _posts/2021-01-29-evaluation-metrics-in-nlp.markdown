---
layout: post
title:  "evaluation metrics in language model"
excerpt: "common evaluation metrics in language model: BLEU, ROUGE, Perplexity"
date:   2021-02-01 17:00:00
mathjax: true
---

### 1.BLEU

**BLEU (bilingual evaluation understudy)** is a metric to evaluate the quality of machine translation between the professional human translation and a machine's output.
Scores are calculated for individual translated segments, by comparing them with a set of good quality reference human translations for one sentence. Then all the scores are
averaged over the whole translated sentences to get the overall quality of translation.

BLEU uses a modified version of **precision** to compare a candidate translation against multiple reference translations. we can use uni-gram, bi-gram, even multi-gram precision as the metric.

One problem with BLEU scores is that they **tend to favor short translations**, which can produce very high precision scores. The **recall** metric is supplementary to the precision to cope with the short translation output.
there is no guarantee that an increase in BLEU score is an indicator of improved translation quality.

### 2.ROUGE

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** is set of metrics used for evaluating automatic summary and machine translation in natural language processing.
It compares the automatically produced result against a or a set of human-produced reference.

- \\( ROUGE-N \\): overlap of \\( N-grams \\) between the model-based output and human-based output.

- \\( ROUGE-L \\): Longest Common Subsequence (LCS) taking into account the sentence-level structure.

- \\( ROUGE-S \\): skip bi-gram based on the co-occurrence statistics.

- \\( ROUGE-SU \\): skip bi-gram and uni-gram.

### 3.Perplexity

In information theory, perplexity is a measurement of how well a probability distribution or probability model predicts a sample. 
It may be used to compare probability models. A low perplexity indicates the probability distribution is good at predicting the sample in the test set.


$$
\begin{equation}
PP (p) = 2^{H(p)} = 2 ^ {- \sum_{x} p(x) \log_{2} p(x)}
\end{equation}
$$

In NLP, perplexity is a way of evaluating language model. A language model is a probability distribution over entire sentences or texts.
The perplexity of the model over the test sentence \\( S \\) is: 

$$
\begin{equation}
perplexity (S) = p(w_1,w_2,\cdots,w_m) ^ {\frac{-1}{m}} = \sqrt[m]{\prod_{i=1}^{m} \frac{1}{p(w_i|w_1,w_2,\cdots,w_{i-1})}}
\end{equation}
$$

### 4.Reference

[BLEU](https://en.wikipedia.org/wiki/BLEU)

[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))

[Perplexity](https://en.wikipedia.org/wiki/Perplexity)


