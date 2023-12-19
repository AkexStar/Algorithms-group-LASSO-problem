import numpy as np
import src.utils as utils

# 近似点梯度法
# 参考 http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_ppa/LASSO_ppa.html

def gl_ProxGD_primal_inner(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: dict = {}):
    opts = utils.optsInnerInit(opts)
    utils.logger.debug(f"--->optsInner:<--- \n{opts}")
    
    mu0 = opts['mu0'] # 目标最小的mu0 由于使用连续化策略，当前的 mu >= mu0
    alpha = opts['alpha0'] # 初始步长

    # 第一次计算
    x = x0
    r = np.matmul(A, x) - b
    g = np.matmul(A.T, r)
    tmp = 0.5 * np.linalg.norm(r, ord='fro') ** 2
    Cval = tmp + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
    f = tmp + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))
    nrmG = np.linalg.norm(x - utils.prox(x - g, mu), ord="fro")
    
    out = utils.outInit()
    f_best = 1e7

    for k in np.arange(opts['maxit_inn']):
        gp = g
        xp = x

        out['g_hist'].append(nrmG)
        out['f_hist'].append(f)
        f_best = np.min([f_best, f])
        utils.logger.debug(f"\tinner iter {k}: fval: {f}, f_best: {f_best}")
        out['f_hist_best'].append(f_best)

        if k > 2 and np.abs(out['f_hist'][k] - out['f_hist'][k-1]) < opts['ftol'] and out['g_hist'][k] < opts['gtol']:
            out['flag'] = True
            break

        # 搜索
        for nls in np.arange(10):
            x = utils.prox(xp - alpha * g, alpha * mu)
            tmp = 0.5 * np.linalg.norm(np.matmul(A, x) - b, ord='fro') ** 2
            tmpf = tmp + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))

            if (tmpf <= Cval - 0.5 * alpha * opts['rhols'] * nrmG ** 2):
                break
            alpha = opts['eta'] * alpha

        g = np.matmul(A.T, np.matmul(A, x) - b)
        f = tmp + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))

        nrmG = np.linalg.norm(x - xp, ord='fro') / alpha
        Qp = opts['Q']
        opts['Q'] = opts['gamma'] * Qp + 1
        Cval = (opts['gamma'] * Qp * Cval + tmpf) / opts['Q']

        # BB
        alpha = utils.BBupdate(x, xp, g, gp, k, alpha)

    out['itr'] = k + 1

    return x, out['itr'], out



def gl_ProxGD_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: dict={}):

    opts = utils.optsOuterInit(opts)
    opts['method'] = gl_ProxGD_primal_inner
    opts['maxit'] = 50
    opts['maxit_inn'] = 30
    opts['gamma'] = 0.85
    x, iter, out = utils.LASSO_group_con(x0, A, b, mu, opts)
    return x, iter, out
