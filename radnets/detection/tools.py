from ..data.preprocess import PREPROCESS, INVERSE_PREPROCESS

def _preprocess(model, X, preprocess):
    if preprocess in ['standardize', 'mean_center']:
        Xhat = PREPROCESS[preprocess](X, model.mu, model.sigma)
    else:
        Xhat = PREPROCESS[preprocess](X)
    return Xhat

def _inv_preprocess(model, X, Xhat, preprocess):
    if preprocess in ['standardize', 'mean_center']:
        Xhat = INVERSE_PREPROCESS[preprocess](Xhat, model.mu, model.sigma)
    elif preprocess in ['normlog']:
        Xhat = INVERSE_PREPROCESS[preprocess](Xhat, X)
    else:
        Xhat = INVERSE_PREPROCESS[preprocess](Xhat)
    return Xhat
