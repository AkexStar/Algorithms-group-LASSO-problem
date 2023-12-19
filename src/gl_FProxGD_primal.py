import numpy as np
import src.utils as utils

# 快速近似点梯度法

def gl_FProxGD_primal_inner(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: dict = {}):
    opts = utils.optsInnerInit(opts)
    utils.logger.debug(f"--->optsInner:<--- \n{opts}")

    mu0 = opts['mu0'] # 目标最小的mu0 由于使用连续化策略，当前的 mu >= mu0
    alpha = opts['alpha0'] # 初始步长
    
    # 第一次计算
    x = x0
    y = x
    xp = x0
    g = np.matmul(A.T, np.matmul(A, y) - b)
    tmp = 0.5 * np.linalg.norm(np.matmul(A, y) - b, ord='fro') ** 2
    Cval = tmp + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
    f = tmp + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))
    nrmG = np.linalg.norm(x - utils.prox(x - g, mu), ord="fro")

    out = utils.outInit()
    f_best = 1e7

    for k in np.arange(opts['maxit_inn']):
        yp = y
        gp = g
        xp = x

        out['g_hist'].append(nrmG)
        out['f_hist'].append(f)
        f_best = np.min([f_best, f])

        out['f_hist_best'].append(f_best)

        if k > 2 and np.abs(out['f_hist'][k] - out['f_hist'][k-1]) < opts['ftol'] and out['g_hist'][k] < opts['gtol']:
            out['flag'] = 1
            break

        for nls in np.arange(10):
            x = utils.prox(y - alpha * g, alpha * mu)
            tmp = 0.5 * np.linalg.norm(np.matmul(A, x) - b, ord='fro') ** 2
            tmpf = tmp + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))

            if (tmpf <= Cval - 0.5 * alpha * opts['rhols'] * nrmG ** 2):
                break
            alpha = opts['eta'] * alpha

        theta = (k - 1) / (k + 2)  # k starts with 0
        y = x + theta * (x - xp)
        r = np.matmul(A, y) - b
        g = np.matmul(A.T, r)
        f = tmp + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))

        # BB
        alpha = utils.BBupdate(y, yp, g, gp, k, alpha)

        nrmG = np.linalg.norm(x - y, ord='fro') / alpha
        Qp = opts['Q']
        opts['Q'] = opts['gamma'] * Qp + 1
        Cval = (opts['gamma'] * Qp * Cval + tmpf) / opts['Q']
        
    out['itr'] = k + 1
    out['flag'] = 1
    return x, out['itr'], out

def gl_FProxGD_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: dict = {}):

    opts = utils.optsOuterInit(opts)
    opts['method'] = gl_FProxGD_primal_inner
    opts['gamma'] = 0.85
    opts['gtol'] = 1e-6
    opts['gtol_init_ratio'] = 1 / opts['gtol']
    opts['ftol'] = 1e-9
    opts['ftol_init_ratio'] = 1e6
    # opts['factor'] = 0.05
    x, iter_, out = utils.LASSO_group_con(x0, A, b, mu, opts)
    return x, iter_, out