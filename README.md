# kalman
Summary code on kalman regression model
The package has been build upon pydlm https://pydlm.github.io/index.html#, generalizing it to multivariate dlm as well as expanding its capability into managing complicated models.
This is an executable file running dynamin linear models using kalman filters.

The main idea is to represent the problem having a system equation and an observation equation, both considered as gaussian distributions having a mean and a variance. The observation equation inherits observation variance from the system as well as from a noise factor which is also a distribution. The main caveat in the model is to be able to estimate the innovation on the next step without having convergence issues. This is the idea of the discount factor introduced by Harrison and West (1999) that gives particular stability to the model.
Using the discount feature the modeler can control how much the current observation will influence the next step prediction.

The multivariate dlm cam calculate multiple observation equations, all sharing the same system variance.
