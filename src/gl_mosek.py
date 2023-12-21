from mosek.fusion import *
import sys
import src.utils as utils

def gl_mosek(x0, A, b, mu, opts={}):
    utils.logger.debug('Note: for mosek, x0 is ignored')
    m, n = A.shape
    l = b.shape[1]
    with Model('gl_mosek') as M:
        M.setLogHandler(sys.stdout)
        A = Matrix.dense(A)
        b = Matrix.dense(b)
        X = M.variable("X", [n, l], Domain.unbounded())
        Y = M.variable("Y", [m, l], Domain.unbounded())
        t1 = M.variable(1)
        ts = M.variable(n, Domain.greaterThan(0.0))
        M.constraint(Expr.sub(Expr.sub(Expr.mul(A, X), b), Y), Domain.equalsTo(0.0))
        M.constraint(Expr.vstack([Expr.add(1, t1), Expr.mul(2, Y.reshape(m * l)), Expr.sub(1, t1)]),
                    Domain.inQCone())
        for i in range(n):
            M.constraint(Expr.vstack([ts.index(i), X.slice([i, 0], [i + 1, l]).reshape(l)]), Domain.inQCone())
        obj = Expr.add(Expr.mul(0.5, t1), Expr.mul(mu, Expr.sum(ts)))
        M.objective('obj', ObjectiveSense.Minimize, obj)
        M.solve()
        with open(utils.cvxLogsName, encoding='utf-8') as f:
            logs = f.read()
        iters = utils.parse_iters(logs, 'MOSEK_OLD')

        utils.logger.debug(f"#######==Solver: MOSEK==#######")
        utils.logger.debug(f"Objective value: {M.primalObjValue()}")
        # utils.logger.debug(f"Status: {M.getSolverIntInfo(' ')}")
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
        out = {'iters': iters, 'fval': M.primalObjValue()}
        return M.getVariable('X').level().reshape(n ,l), iters_N, out

