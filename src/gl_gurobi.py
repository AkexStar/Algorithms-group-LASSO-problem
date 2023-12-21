import gurobipy as gp
import numpy as np
import src.utils as utils

def gl_gurobi(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts={}):
    m, n = A.shape
    l = b.shape[1]
    model = gp.Model()
    X = model.addMVar((n, l), lb=-gp.GRB.INFINITY)
    # X = model.addMVar((n, l), lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")
    X.start = x0
    Y = model.addMVar((m, l), lb=-gp.GRB.INFINITY)
    ts = model.addMVar(n, lb=0.0)
    for j in range(l):
        model.addConstr(A @ X[:, j] - b[:, j] == Y[:, j])
    for i in range(n):
        model.addConstr(X[i, :] @ X[i, :] <= ts[i] * ts[i])
    model.setObjective(0.5 * sum([Y[:, j] @ Y[:, j] for j in range(l)]) + mu * sum(ts), gp.GRB.MINIMIZE)
    model.optimize()
    with open(utils.cvxLogsName, 'r', encoding='utf-8') as f:
        logs = f.read()
    iters = utils.parse_iters(logs, 'GUROBI')

    utils.logger.debug(f"#######==Solver: GUROBIPY==#######")
    utils.logger.debug(f"Objective value: {model.objVal}")
    utils.logger.debug(f"Status: {model.status}")
    # utils.logger.debug(f"Solver status: {model.solver_stats}")
    utils.logger.debug(f"#######==CVXPY's Logs:==#######\n{logs}")
    utils.logger.debug(f"#######==END of Logs:==#######")
    utils.logger.debug(f"iters after parse:\n{iters}")

    if len(iters) == 0:
        utils.logger.error(f"求解器cvx(MOSEK)的记录迭代次数为0。需要检查stdout重定向情况！")
        iters_N = -1
    else:
        iters_N = len(iters)
    # 最优解, 迭代次数, 情况输出out={iters: 每次迭代目标函数值情况[(iter, fval), ...]; fval: 最终目标函数值}
    out = {'iters': iters, 'fval': model.objVal}
    return X.x, iters_N, out
