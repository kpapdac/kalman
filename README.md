# kalman
Summary code on kalman regression model
The package has been build upon pydlm https://pydlm.github.io/index.html#, generalizing it to multivariate dlm as well as expanding its capability into managing complicated models.
This is an executable file running dynamin linear models using kalman filters.

The main idea is to represent the problem having a system equation and an observation equation, both considered as gaussian distributions having a mean and a variance. The observation equation inherits observation variance from the system as well as from a noise factor which is also a distribution. The main caveat in the model is to be able to estimate the innovation on the next step without having convergence issues. This is the idea of the discount factor introduced by Harrison and West (1999) that gives particular stability to the model.
Using the discount feature the modeler can control how much the current observation will influence the next step prediction.

The multivariate dlm cam calculate multiple observation equations, all sharing the same system variance.

The theory of kalman filters can be summarised as follows.

At each time t system and observations are defined from the equations

Observation: Y(t)= μ(t) + ν(t), ν(t) ∼ N[0, V(t)]

System: μ(t)= μ(t−1) + ω(t), ω(t) ∼ N[0, W(t)]

Initial information: (μ0 |D0) ∼ N[m(0), C(0)]

The next step prediction is given by the following set of recursive relationships

(a) Posterior for μt−1 : (μt−1 | Dt−1) ∼ N[m(t−1), C(t−1)]

(b) Prior for μt : (μt | Dt−1) ∼ N[m(t−1), R(t)],
where R(t) = C(t−1) + W(t).

(c) 1-step forecast: (Yt | Dt−1) ∼ N[f(t), Q(t)],
where f(t) = m(t−1) and Q(t) = R(t) + V(t).

(d) Posterior for μt : (μt | Dt) ∼ N[m(t), C(t)],
with m(t) = m(t−1) + A(t) * e(t) and C(t) = A(t) * V(t),
where A(t) = R(t) /Q(t) , and et = Y(t) − f(t).
