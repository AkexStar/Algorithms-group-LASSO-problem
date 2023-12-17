from mosek.fusion import *
import sys
import code.utils as utils

def gl_mosek(x0, A, b, mu, opts={}):
    utils.logger.info('Note: for mosek, x0 is ignored')
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

        utils.logger.info(f"#######==Solver: cvx(MOSEK)==#######")
        utils.logger.info(f"Objective value: {M.primalObjValue()}")
        # utils.logger.info(f"Status: {M.getSolverIntInfo(' ')}")
        # utils.logger.info(f"Solver status: {model.solver_stats}")
        utils.logger.info(f"#######==CVXPY's Logs:==#######\n{logs}")
        utils.logger.info(f"#######==END of Logs:==#######")
        utils.logger.info(f"iters after parse:\n{iters}")
        # 最优解，迭代次数，{iters每次迭代目标函数值情况，cpu_time求解时间；obj目标函数值}
        out = {'iters': iters, 'obj': M.primalObjValue()}
        return M.getVariable('X').level().reshape(n ,l), len(iters), out

