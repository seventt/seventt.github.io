---
layout: post
title:  "pre-training model in NLP"
excerpt: "from word embeddings to pre-training model based on large-scale data set to better capture and represent the contextual information including Word2Vec, ELMo, Transformer, GPT, BERT and ERNIE"
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

<div class="imgcap">
<img src="/assets/bert/word2vec-ret.png">
<div class="thecap">The illustration of word embedding for some kind of words .</div>
</div>

But, there are some problems that restricts the performance of Word2Vec.

- One is: there is one **fixed vector representation** for one word, so it is not suitable for **synonym word (polysemy)**,
- another is: Word2Vec takes less contextual information of current word into consideration, the representation is a little **shallow**,

### 2.ELMo

**ELMo (Embeddings from Language Models)** is a deep contextualized word representation that models both complex characteristics of word use (e.g., syntax and semantics), and how these uses vary across linguistic contexts (i.e., to model polysemy). These word vectors are learned functions of the internal states of a deep bidirectional language model (**biLM**), 
which is pre-trained on a large text corpus. They can be easily added to existing models and significantly improve the state of the art across a broad range of challenging NLP problems, including question answering, textual entailment and sentiment analysis.

The architecture of ELMo is:

<div class="imgcap">
<img src="/assets/bert/elmo-network.jpg">
<div class="thecap">The architecture of ELMo.</div>
</div>

The characteristics of ELMo representations are;

- **Deep**: The word representations combine all layers of a deep pre-trained neural network.

- **Contextual**: The representation for each word depends on the entire context in which it is used. Instead of using a fixed embedding for each word, ELMo looks at the entire sentence before assigning each word in it an embedding

ELMo comes up with the contextualized embedding through grouping together the hidden states (and initial embedding) in a certain way (concatenation followed by weighted summation).

<div class="imgcap">
<img src="/assets/bert/elmo-embedding.png">
<div class="thecap">The illustration of finalword embedding based on ELMo.</div>
</div>

the dynamic process to get deep contextualized word embedding is as following:

<div class="imgcap">
<img src="/assets/bert/elmo.gif">
<div class="thecap">The illustration of ELMo.</div>
</div>

### 3.Transformer

The Transformer model based on encoder-decoder architecture was proposed in [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf).

<div class="imgcap">
<img src="/assets/bert/transformer-architecture.png">
<div class="thecap">The model architecture of Transformer.</div>
</div>

The sequential nature of Recurrent models (RNN, LSTM) causes that the parallel computation can't be exploited within the training examples,
and the memory constraints limit the batching across training examples as the length of sequence becomes longer. 

The Transformer substitutes the traditional attention mechanism using recurrent or convolutional neural networks in encoder and decoder with 
the simple **multi-head self-attention mechanism** in the task - machine translation.
It allows for significantly **more parallelization** for the single training example with the help of multi-headed self-attention mechanism, in addition, it can also learn **long-range dependency**.

<div class="imgcap">
<img src="/assets/bert/self-attention.png">
<div class="thecap">The computation flow of single self-attention.</div>
</div>

The Transformer makes use of multi-headed self-attention to improve the performance of attention layer by concatenation of multi self-attention output result. Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions. In the decoder, We only need to keep leftward information flow of current input word by masking out all future token values in the input of the softmax which correspond to illegal connections using the **masked self-attention**.

Since our model contains no recurrence and no convolution, in order for the model to make use of the
order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.
we add **positional encoding** to the input embeddings at the bottoms of the encoder and decoder stacks.

### 4.GPT

**GPT (Generative Pre-trained Transformer)** is a unsupervised pre-training model trained on enormous, diverse and unlabelled corpus of text to further improve the natural language understanding performance in NLP, 
followed by discriminative fine-tuning on each specific downstream task.

It explores a semi-supervised approach for language understanding tasks using a **combination of unsupervised pre-training and supervised fine-tuning**. 
Its goal is to learn a universal representation that transfers with little adaptation to a wide range of tasks such as text classification, question answering, sequence tagging, semantic similarity and so on. 
This alleviates mostly the dependency on supervised learning with labelled data.

It employs a two-stage training procedure:
- First, for unsupervised pre-training, we use a language modeling objective on the unlabeled data to learn the initial parameters of a neural network model based on **Transformer Decoder Block** architecture. 
- Subsequently, for supervised fine-tuning stage, we adapt these parameters to a target task using the corresponding supervised objective.

The process of pre-training is to predict the word based on its previous words and their contextual representation for one sentence same as the decoder process of Transformer.
The way these models actually work is that after each token is produced, that token is added to the sequence of inputs. And that new sequence becomes the input to the model in its next step. This is an idea called **auto-regression**.

The masked self-attention in decoder is:

<div class="imgcap">
<img src="/assets/bert/masked-self-attention.png">
<div class="thecap">The illustration of masked self-attention in the decoder.</div>
</div>

The computation of masked self-attention in decoder is:

<div class="imgcap">
<img src="/assets/bert/masked-self-attention-computation.png">
<div class="thecap">The illustration of computation of masked self-attention in the decoder.</div>
</div>

The process of fine-tuning is as follows:

<div class="imgcap">
<img src="/assets/bert/gpt.png">
<div class="thecap">The fine-tuning process of GPT.</div>
</div>

### 5.BERT

**BERT (Bidirectional Encoder Representation of Transformer)** is based on the **Transformer Encoder Block**.

### 6.ERNIE





### 7.Reference

[Word2vec](https://www.tensorflow.org/tutorials/text/word2vec)

[ELMo](https://allennlp.org/elmo)

[BERT,ELMo](http://jalammar.github.io/illustrated-bert/)

[Transformer](https://jalammar.github.io/illustrated-transformer/)

[official implementation of Transformer based on tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)

[implementation of Transformer based on Tensorflow](https://github.com/tensorflow/models/tree/master/official/nlp/transformer)

[GPT-2](http://jalammar.github.io/illustrated-gpt2/)

[huggingface-transformers](https://github.com/huggingface/transformers)




