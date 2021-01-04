---
layout: post
comments: true
title:  "optimization algorithm for neural network"
excerpt: "A brief introduction of gradient descent optimization algorithms"
date:   2021-01-04 15:00:00
mathjax: true
---

### SGD

Stochastic Gradient Descent is a simple optimization method to minimize the cost function.

$$
\begin{equation}
\theta_{i+1} = \theta_i - \eta g_t
\end{equation}
$$

The disadvantage of SGD is the slow speed of convergence and oscillation at saddle point

### Momentum

The next update is not only dependent on the current gradient but also the previous gradients.

$$
\begin{align}
m_{i+1} = \gamma m_i + \eta g_t \\
\theta_{i+1} = \theta_i - m_{i+1}
\end{align}
$$

### AdaGrad

It adapts the learning rate to the parameters, performing smaller updates
(i.e. low learning rates) for parameters associated with frequently occurring features, and larger updates (i.e. high learning rates) for parameters associated with infrequent features. For this reason, it is well-suited for dealing with sparse data.

$$
\begin{align}
\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii}+\epsilon}} g_t
\end{align}
$$

the learning rate is scaled up by the sum over square of past gradients. this result in the radical diminish of learning rate.

### RMSProp

RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients.

$$
\begin{align}
G_{t,ii} = \gamma G_{t-1,ii} + (1-\gamma) g_t^2 \\
\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii}+\epsilon}} g_t
\end{align}
$$

### Adam

Adaptive Moment Estimation (Adam) is another method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients 
like AdaGrad and RMSprop, Adam also keeps an exponentially decaying average of past gradients similar to momentum.

$$
\begin{equation}
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 
\end{equation}
$$

Generally, \\( \beta_1\\) is 0.9 and \\( \beta_2\\) is 0.999 and \\( \epsilon\\) is 10e-8.



