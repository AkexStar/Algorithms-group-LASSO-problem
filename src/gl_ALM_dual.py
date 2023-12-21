import numpy as np
import src.utils as utils

# 增广拉格朗日函数法
# 参考 http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/lect/16-lect-alm-pku.pdf

def updateZ(ref, mu):
    norm = np.linalg.norm(ref, axis=1, keepdims=True)
    norm[norm < mu] = mu
    return ref * (mu / norm)

def gl_ALM_dual(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts:dict = {}):
    opts = utils.ALM_dual_optsInit(opts)    
    utils.logger.info(f"optsOuter: \n{opts}")
    m, n = A.shape
    _, l = b.shape
    out = utils.outInit()
    out['prim_hist'] = []
    out['dual_hist'] = []
    x = x0
    
    sigma = opts['sigma']   # 二次罚函数系数
    inv = np.linalg.inv(np.eye(m) + sigma * A @ A.T)
    z = np.zeros((n, l))

    for k1 in range(opts['maxit']):
        for k2 in range(opts['maxit_inn']):
            y = inv @ (A @ x - sigma * A @ z  - b)
            zp = z
            z = updateZ(x / sigma - A.T @ y, mu)
            inner_gap = np.linalg.norm(z - zp, 'fro')
            out['itr_inn'] += 1
            if inner_gap < opts['thre_inn']:
                break

        x = x - sigma * (A.T @ y + z)
        # f = 0.5 * np.linalg.norm(A @ x - b, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
        f = utils.objFun(x, A, b, mu)
        f_dual = 0.5 * np.linalg.norm(y, ord='fro') ** 2 - np.sum(y * b)
        utils.logger.debug(f"f - f_dual: {f - f_dual}")
        out['prim_hist'].append(f)
        out['dual_hist'].append(f_dual)

        if np.linalg.norm(A.T @ y + z) < opts['thre']:
            break
    
    out['itr'] = k1 + 1
    out['fval'] = out['prim_hist'][-1]
    utils.logger.debug(f"ALM_dual: itr: {out['itr']}, fval: {out['fval']}")
    utils.logger.debug(f"ALM_dual: itr_inn: {out['itr_inn']}")
    utils.logger.debug(f"ALM_dual: len(out['prim_hist']): {len(out['prim_hist'])}")
    out['iters'] = zip(range(out['itr']), out['prim_hist'])#, out['dual_hist'])
    # out['iters'] = zip(range(out['itr']), out['dual_hist'])
        
    return x, out['itr'], out