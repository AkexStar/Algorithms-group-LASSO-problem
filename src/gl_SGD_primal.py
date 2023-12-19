import numpy as np
import src.utils as utils

# 次梯度法求解原始问题（使用连续化策略）
# 参考 http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_subgrad/LASSO_subgrad_inn.html
# 参考 http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_subgrad/demo_cont.html

def gl_SGD_primal_inner(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: dict = {}):
    opts = utils.optsInnerInit(opts)
    utils.logger.debug(f"--->optsInner:<--- \n{opts}")

    mu0 = opts['mu0'] # 目标最小的mu0 由于使用连续化策略，当前的 mu >= mu0
    alpha = opts['alpha0'] # 初始步长

    # 第一次计算
    x = x0
    r = np.matmul(A, x) - b
    g = np.matmul(A.T, r)
    norm_x = np.linalg.norm(x, axis=1).reshape((-1, 1))
    sub_g = x / ((norm_x <= 1e-6) + norm_x)
    sub_g = sub_g * mu + g
    nrmG = np.linalg.norm(g, ord="fro")
    tmp = 0.5 * np.linalg.norm(r, ord='fro') ** 2
    Cval = tmp + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
    f = tmp + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))
    utils.logger.debug(f"inner fval: {f}")

    out = utils.outInit()
    # f_best = 1e7
    
    for k in np.arange(opts['maxit_inn']):
        gp = g
        xp = x
        out['g_hist'].append(nrmG)
        out['f_hist'].append(f)
        # f_best = np.min([f_best, f])
        # utils.logger.debug(f"\tinner iter {k}: fval: {f}, f_best: {f_best}")
        # out['f_hist_best'].append(f_best)

        if k > 2 and np.abs(out['f_hist'][k] - out['f_hist'][k-1]) < opts['ftol']:
            out['flag'] = True
            break
        
        # 搜索
        for nls in np.arange(10):
            x = xp - alpha * sub_g
            tmp = 0.5 * np.linalg.norm(np.matmul(A, x) - b, ord='fro') ** 2
            tmpf = tmp + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))

            if (tmpf <= Cval - 0.5 * alpha * opts['rhols'] * nrmG ** 2):
                break
            alpha = opts['eta'] * alpha

        g = np.matmul(A.T, np.matmul(A, x) - b)
        norm_x = np.linalg.norm(x, axis=1).reshape((-1, 1))
        sub_g = x / ((norm_x <= 1e-6) + norm_x)
        sub_g = sub_g * mu + g

        f = tmp + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))

        nrmG = np.linalg.norm(g, ord='fro')
        Qp = opts['Q']
        opts['Q'] = opts['gamma'] * Qp + 1
        Cval = (opts['gamma'] * Qp * Cval + tmpf) / opts['Q']

        # BB
        alpha = utils.BBupdate(x, xp, g, gp, k, alpha)

    out['itr'] = k + 1
    # utils.debug.info(f"f_hist_best: {out['f_hist_best']}")
    return x, out['itr'], out

def gl_SGD_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts={}):
    """
    - Subgradient method for the primal problem.
    - minimize   (1/2)*||Ax-b||_2^2 + mu*||x||_{1,2}
    - A is m x n, b is m x l, x is n x l
    """
    opts = utils.optsOuterInit(opts)
    opts['method'] = gl_SGD_primal_inner
    opts['maxit'] = 30
    opts['maxit_inn'] = 130
    x, iter, out = utils.LASSO_group_con(x0, A, b, mu, opts)
    return x, iter, out
    
    