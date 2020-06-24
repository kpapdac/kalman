# kalman
Summary code on kalman regression model
The package has been build upon pydlm https://pydlm.github.io/index.html#, generalizing it to multivariate dlm as well as expanding its capability into managing complicated models.
This is an executable file running dynamin linear models using kalman filters.

The main idea is to represent the problem having a system equation and an observation equation, both considered as gaussian distributions having a mean and a variance. The observation equation inherits observation variance from the system as well as from a noise factor which is also a distribution. The main caveat in the model is to be able to estimate the innovation on the next step without having convergence issues. This is the idea of the discount factor introduced by Harrison and West (1999) that gives particular stability to the model.
Using the discount feature the modeler can control how much the current observation will influence the next step prediction.

The multivariate dlm cam calculate multiple observation equations, all sharing the same system variance.

The theory of kalman filters can be summarised as follows.

At each time t system and observations are defined from the equations

Observation: Yt= μt + νt, νt ∼ N[0, Vt]

System: μt= μt−1 + ωt, ωt ∼ N[0, Wt]

Initial information: (μ0 |D0) ∼ N[m0, C0]

The next step prediction is given by the following set of recursive relationships

(a) Posterior for μt−1 : (μt−1 | Dt−1) ∼ N[mt−1, Ct−1]

(b) Prior for μt : (μt | Dt−1) ∼ N[mt−1, Rt],
where Rt = Ct−1 + Wt.

(c) 1-step forecast: (Yt | Dt−1) ∼ N[ft, Qt],
where ft = mt−1 and Qt = Rt + Vt.

(d) Posterior for μt : (μt | Dt) ∼ N[mt, Ct],
with mt = mt−1 + Atet and Ct = AtVt,
where At = Rt /Qt , and et = Yt − ft.
