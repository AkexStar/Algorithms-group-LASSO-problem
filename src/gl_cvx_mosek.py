import numpy as np
import cvxpy as cp
import src.utils as utils

def gl_cvx_mosek(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts={}):
    X = cp.Variable(shape=(A.shape[1], b.shape[1]))
    X.value = x0
    # mu = cp.Parameter(nonneg=True, value=mu)
    # objective = cp.Minimize(0.5 * cp.sum_squares(A @ X - b) + mu * cp.mixed_norm(X, 2, 1))
    objective = cp.Minimize(0.5 * cp.square(cp.norm(A @ X - b, 'fro')) + mu * cp.sum(cp.norm(X, 2, 1)))
    problem = cp.Problem(objective)
    problem.solve(solver=cp.MOSEK, verbose=True)
    with open(utils.cvxLogsName, 'r', encoding='utf-8') as f:
        logs = f.read()
    iters = utils.parse_iters(logs, 'MOSEK')

    utils.logger.debug(f"#######==Solver: cvx(MOSEK)==#######")
    utils.logger.debug(f"Objective value: {problem.value}")
    utils.logger.debug(f"Status: {problem.status}")
    utils.logger.debug(f"Solver status: {problem.solver_stats}")
    utils.logger.debug(f"#######==CVXPY's Logs:==#######\n{logs}")
    utils.logger.debug(f"#######==END of Logs:==#######")
    utils.logger.debug(f"iters after parse:\n{iters}")
    
    if len(iters) == 0:
        utils.logger.error(f"求解器cvx(MOSEK)的记录迭代次数为0。需要检查stdout重定向情况！")
        iters_N = -1
    else:
        iters_N = len(iters)
    out = {'iters': iters, 'fval': problem.value}
    # 最优解, 迭代次数, 情况输出out={iters: 每次迭代目标函数值情况[(iter, fval), ...]; fval: 最终目标函数值}
    return X.value, iters_N, out