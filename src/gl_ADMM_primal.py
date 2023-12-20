import numpy as np
import src.utils as utils

# 交换方向乘子法（原问题）
# 参考 http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_admm/demo_admm.html
# 参考 http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_admm/LASSO_admm_primal.html

def gl_ADMM_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts:dict = {}):
    opts = utils.ADMM_primal_optsInit(opts)
    _, n = A.shape
    _, l = b.shape
    out = utils.outInit()
    out['prim_hist'] = []

    x = x0
    y = x0
    z = np.zeros((n, l))

    sigma = opts['sigma']   # 二次罚函数系数
    inv = np.linalg.inv(sigma * np.eye(n) + A.T @ A) # 由于罚因子在算法的迭代过程中未变化，事先缓存 Cholesky 分解可以加速迭代过程。
    ATb = A.T @ b

    for k in range(opts['maxit']):
        x = inv @ (sigma * y + ATb - z)
        y0 = y
        y = utils.prox(x + z / sigma, mu / sigma)
        z = z + sigma * (x - y)

        primal_sat = np.linalg.norm(x - y)
        dual_sat = np.linalg.norm(y0 - y)
        # f = 0.5 * np.linalg.norm(A @ x - b, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
        f = utils.objFun(x, A, b, mu)
        out['prim_hist'].append(f)
        # out['itr'] += 1

        if primal_sat < opts['thre'] and dual_sat < opts['thre']:
            break
    
    # utils.logger.info(f"k: {k}")
    out['itr'] = k + 1
    out['fval'] = f
    utils.logger.info(f"ADMM_primal: itr: {out['itr']}, fval: {out['fval']}")
    utils.logger.info(f"ADMM_primal: len(out['prim_hist']): {len(out['prim_hist'])}")
    out['iters'] = zip(range(out['itr']), out['prim_hist'])

    return x, out['itr'], out


