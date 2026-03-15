import numpy as np
from hmmlearn import hmm

ev_model = hmm.MultinomialHMM(n_components=2)

ev_model.startprob_ = np.array([1.0,0.0])

ev_model.transmat_ = np.array([
[0.6,0.4],
[0.2,0.8]
])

ev_model.emissionprob_ = np.array([
[0.7,0.3],
[0.1,0.9]
])


okul_model = hmm.MultinomialHMM(n_components=2)

okul_model.startprob_ = np.array([1.0,0.0])

okul_model.transmat_ = np.array([
[0.5,0.5],
[0.3,0.7]
])

okul_model.emissionprob_ = np.array([
[0.6,0.4],
[0.3,0.7]
])


def classify(obs):

    obs=np.array(obs).reshape(-1,1)

    ev_score=ev_model.score(obs)
    okul_score=okul_model.score(obs)

    if ev_score>okul_score:
        return "EV"
    else:
        return "OKUL"


test=[0,1]

print(classify(test))
