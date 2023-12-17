import numpy as np
import cvxpy as cp
import utils

def gl_cvx_mosek(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts={}):
    X = cp.Variable(shape=(A.shape[1], b.shape[1]))
    X.value = x0
    # mu = cp.Parameter(nonneg=True, value=mu)
    # objective = cp.Minimize(0.5 * cp.sum_squares(A @ X - b) + mu * cp.mixed_norm(X, 2, 1))
    objective = cp.Minimize(0.5 * cp.square(cp.norm(A @ X - b, 'fro')) + mu * cp.sum(cp.norm(X, 2, 1)))
    problem = cp.Problem(objective)
    problem.solve(solver=cp.MOSEK, verbose=True)
    with open('./logs/gl_cvx.log', 'r', encoding='utf-8') as f:
        logs = f.read()
    iters = utils.parse_iters(logs, 'MOSEK')

    utils.logger.info(f"#######==Solver: cvx(MOSEK)==#######")
    utils.logger.info(f"Objective value: {problem.value}")
    utils.logger.info(f"Status: {problem.status}")
    utils.logger.info(f"Solver status: {problem.solver_stats}")
    utils.logger.info(f"#######==CVXPY's Logs:==#######\n{logs}")
    utils.logger.info(f"#######==END of Logs:==#######")
    utils.logger.info(f"iters after parse:\n{iters}")
    # 最优解，迭代次数，{iters每次迭代目标函数值情况，cpu_time求解时间；obj目标函数值}
    out = {'iters': iters, 'obj': problem.value}
    return X.value, len(iters), out