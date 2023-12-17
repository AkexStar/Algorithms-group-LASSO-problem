import gurobipy as gp
import numpy as np
import utils

def gl_gurobi(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts={}):
    m, n = A.shape
    l = b.shape[1]
    model = gp.Model()
    X = model.addMVar((n, l), lb=-gp.GRB.INFINITY)
    # X = model.addMVar((n, l), lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")
    X.start = x0
    Y = model.addMVar((m, l), lb=-gp.GRB.INFINITY)
    ts = model.addMVar(n)
    for j in range(l):
        model.addConstr(A @ X[:, j] - b[:, j] == Y[:, j])
    for i in range(n):
        model.addConstr(X[i, :] @ X[i, :] <= ts[i] * ts[i])
    model.setObjective(0.5 * sum([Y[:, j] @ Y[:, j] for j in range(l)]) + mu * sum(ts), gp.GRB.MINIMIZE)
    model.optimize()
    with open('./logs/gl_cvx.log', 'r', encoding='utf-8') as f:
        logs = f.read()
    iters = utils.parse_iters(logs, 'GUROBI')

    utils.logger.info(f"#######==Solver: cvx(MOSEK)==#######")
    utils.logger.info(f"Objective value: {model.objVal}")
    utils.logger.info(f"Status: {model.status}")
    # utils.logger.info(f"Solver status: {model.solver_stats}")
    utils.logger.info(f"#######==CVXPY's Logs:==#######\n{logs}")
    utils.logger.info(f"#######==END of Logs:==#######")
    utils.logger.info(f"iters after parse:\n{iters}")
    # 最优解，迭代次数，{iters每次迭代目标函数值情况，cpu_time求解时间；obj目标函数值}
    out = {'iters': iters, 'obj': model.objVal}
    return X.x, len(iters), out
