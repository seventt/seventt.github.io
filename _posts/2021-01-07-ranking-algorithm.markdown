---
layout: post
title:  "ranking algorithm"
excerpt: "Ranking methods including PageRank, BM25 and Learning to Rank (L2R)"
date:   2021-01-08 17:00:00
mathjax: true
---

### 1.PageRank

PageRank (PR) is a way of measuring the importance of website pages by counting the number and quality of inbound links of a page to determine a rough estimate of how important the website is. 
The underlying assumption is that more important websites are likely to receive more links from other websites. 

The computation of rank of pages is described as the following equation:

$$
\begin{equation}
PR(p_i) = \frac{1-d}{N} + d \sum_{p_j \in M(p_i)} \frac{PR(p_j)}{L(p_j)} \\
M(p_i): \text{the pages that link to} \; p_i \\
L(p_j): \text{the number of outbound links on page } p_j \\
N:\text{the number of total pages} \\
d:\text{damping factor (continue to click on this page)} 
\end{equation}
$$

The computation of matrix form:

$$
\begin{equation}
\boldsymbol{R} = 
\begin{bmatrix}
PR(p_1) \\
PR(p_2) \\
\vdots \\
PR(p_N)
\end{bmatrix}
=
\begin{bmatrix}
(1-d)/N \\
(1-d)/N \\
\vdots \\
(1-d)/N
\end{bmatrix}
+d
\begin{bmatrix}
l(p_1,p_1) & l(p_1,p_2) & \dots & l(p_1,p_N) \\
l(p_2,p_1) & l(p_2,p_2) & \dots & l(p_2,p_N) \\
\vdots & \vdots & l(p_i,p_j) & \vdots \\
l(p_N,p_1) & l(p_N,p_2) & \dots & l(p_N,p_N) 
\end{bmatrix}
\boldsymbol{R}
\end{equation}
$$

the initial probability of all pages is set to \\( \frac{1}{N}\\) 

### 2.BM25

In information retrieval, BM25 (BM is best matching) is a ranking function used by search engines 
to estimate the relevance of documents to a given search query.

$$
\begin{equation}
score(D,Q) = \sum_{1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)} {f(q_i, D) + k_1 \cdot (1-b+b\cdot \frac{\mid D \mid}{avgdl})} \\
\text{inverse document frequency is to weight the words that common words have less weight:} \quad IDF(q_i) = \log (\frac{N-n(q_i)+0.5}{n(q_i)+0.5} + 1) \\ 
n(q_i): \text{the number of documents containing query} \, q_i\\
f(q_i, D): \text{term frequency:}\\
k_1 \in (1.2,2.0):\text{constraints on the term frequency} \\
b=0.75:\text{constraints on the document length}
\end{equation}
$$

### 3.Learning To Rank

#### 3.1.Pointwise

For each recalled document of specific query, we need judge and label if the document is related to the query. Then we can model the relationship between query/document and their label by different machine learning classification models.
The drawback is that plenty of human labelling is expensive.

#### 3.2.Pairwise

For the pair \\( (query_k, doc_i, doc_j)\\), the label is 1 if the relevance between \\( query_k \\) and \\( doc_i \\) is closer than the relevance between \\( query_k \\) and \\( doc_j \\), otherwise, the label is 0, which can be set by the search log
This means considers the relevant importance of different documents related to the query. Typical model is RankNet, LambdaRank and LambdaMART.

**RankNet:** the output of model is \\( s_i, s_j\\) and  \\(P(rank(i) > rank(j)) \\) is the probability of the relevance of \\( (query_k, doc_i)\\) is larger compared to \\( (query_k, doc_j)\\).

$$
\begin{equation}
P(rank(i) > rank(j)) = \frac{1}{1+e^{-\sigma(s_i-s_j)}}
\end{equation}
$$

The loss function is described as the following equation:

$$
\begin{equation}
L_{i,j} = - \log \frac{1}{1+e^{-\sigma(s_i-s_j)}}
\end{equation}
$$

derivative of loss of each pair with respect to weights:

$$
\begin{equation}
\frac{\partial L_{i,j}}{\partial \theta_k} = \frac{-1}{1+e^{s_i-s_j}} (\frac{\partial s_i}{\partial \theta_k} - \frac{\partial s_j}{\partial \theta_k} ) = \lambda_{i,j}(\frac{\partial s_i}{\partial \theta_k} - \frac{\partial s_j}{\partial \theta_k} )
\end{equation}
$$

and total derivative of loss of batch pairs with respect to weights:

$$
\begin{equation}
\frac{\partial L}{\partial \theta_k} = \sum_{(i,j)\in D}\frac{\partial L_{i,j}}{\partial \theta_k} 
 = \sum_{(i,j)\in D}\lambda_{i,j}(\frac{\partial s_i}{\partial \theta_k} - \frac{\partial s_j}{\partial \theta_k} )
\end{equation}
$$

the aim of gradient update is to move the weights to increase the score of the item with the higher rank and decrease the score of the item with the lower rank with a certain weighting factor.
the smaller the difference of the higher ranking item compared to the lower ranking item, the larger the weight as described in the following picture

<div class="imgcap">
<img src="/assets/ranking.png">
<div class="thecap">THe relationship between weight and the difference in scores.</div>
</div>

The disadvantege is: Swapping 1st and 10th is essentially equivalent to swapping 101st and 110th place because the only thing that matters is that the rankings change by 10 places (as long as the difference in scores are the same).

**LambdaRank:** weight the gradient for the pairwise loss by a factor that incorporates the importance of getting the pairwise ranking correct. This implies that more important the document is to the query, more weight the document gets if it is in a wrong order.
Generally, it can be evaluated by the metric NDCG.

$$
\begin{equation}
\lambda_{i,j}= \frac{-1}{1+e^{s_i-s_j}} \mid \Delta(i,j) \mid 
\quad \quad
\mid \Delta(i,j) \mid:\text{a penalty corresponding to how "bad" it is to rank items i  and j  in the wrong order. }
\end{equation}
$$

#### 3.3.Listwise

This method is to label the relative importance of all the documents to the specific query and then to model this relationship. It is harder to converge.

### 4.Evaluation Metric

There are some metrics to evaluate the performance of the ranking model.

#### 4.1.MAP

**Mean Average Precision(MAP):** it only takes the relevance of document into consideration.

$$
\begin{equation}
MAP=\frac{\sum_{q=1}^{Q} AveP(q)}{Q} \newline
AveP = \frac{\sum_{i=1}^{n} P(K) rel(K)}{\text{number of relevant documents}}
\end{equation}
$$

#### 4.2.NDCG

**Normalized Discounted Cumulative Gain(NDCG):** It combines the levels and the importance of the relevance of documents. If the most relevant documents gets the lower rank, the value of NDCG will be smaller. The documents make different contribution to the final NDCG value
according to their actual ranking score.

$$
\begin{equation}
NDCG@T = \frac{DCG@T}{IDCG@T}
\end{equation}
$$

**Discounted Cumulative Gain(DCG):**

$$
\begin{equation}
DCG@T = \sum_{i=1}^{T} \frac{2^{l_i}-1}{\log(1+i)} \newline
T: \text{the former T predicted results} \newline
l_i: \text{the actual score of document}\newline
i: \text{the position of document in the predicted results}
\end{equation}
$$

**Ideal Discounted Cumulative Gain(IDCG):** the value of DCG when the model has the optimal ranking. 

#### 4.3.MRR

**Mean Reciprocal Rank(MRR):** the reciprocal of position of the most relevant document to the query.

### 5.Reference

[PageRank](https://en.wikipedia.org/wiki/PageRank)
[BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
[NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)


