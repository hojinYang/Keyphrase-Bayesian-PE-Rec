import numpy as np


###MLE or MAP estimation###

def get_point_estimate(X, Y, lam=0):
    dim = X.shape[1]
    pre_inv = X.T @ X + lam * np.identity(dim)
    #lam = prec_w/prec_y
    inverse = np.linalg.inv(pre_inv)
    W = inverse @ X.T @ Y
    return W


### Sequential Bayesian Updating ###

def update_posterior(x, y, S_0, m_0, prec_y):
    S_0_inv = np.linalg.inv(S_0)
    #S_1 = np.linalg.inv(S_0_inv +prec_y * x @ x.T)    
    #print(np.swapaxes(x,-2,-1).shape)
    S_1 = np.linalg.inv(S_0_inv +prec_y * np.matmul(x,np.swapaxes(x,-2,-1)))
    m_1 = S_1 @ (S_0_inv @ m_0 + prec_y * y * x)
    #print(m_1.shape)
    return S_1, m_1

def get_predictive_dist(x_pred, S, m, prec_y):
    pred_means = m.T @ x_pred
    pred_means = pred_means.flatten()
    pred_vars = np.sum(1/prec_y + x_pred.T @ S * x_pred.T, axis=1)

    return pred_vars, pred_means


### Bayesian Linear Regression with batch ###

def bayesian_linear_reg(prec_W, prec_y, X, y):
    dim = X.shape[1]
    cov_inv = prec_y*(X.T @ X) + prec_W*np.identity(dim)
    cov = np.linalg.inv(cov_inv)
    mean = (prec_y*cov) @ X.T @ y

    return cov, mean

def get_user_dist(uid, r_ui, I, prec_W, prec_y):
    return bayesian_linear_reg(prec_W, prec_y, I, r_ui[uid].T)

