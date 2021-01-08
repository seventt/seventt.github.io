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

#### 3.2.Pairwise

#### 3.3.Listwise

### 4.Reference

[PageRank](https://en.wikipedia.org/wiki/PageRank)

[BM25](https://en.wikipedia.org/wiki/Okapi_BM25)


