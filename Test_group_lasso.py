import code.utils as utils
import time
import importlib
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Test_group_lasso', description='测试不同的求解器进行group lasso求解')
    parser.add_argument('--solvers', '-S', nargs='+', default=['gl_cvx_gurobi', 'gl_cvx_mosek'], help='指定求解器名称, 输入`all` 可以测试本项目中所有的求解器函数。默认填充为 `[\'gl_cvx_gurobi\', \'gl_cvx_mosek\']` 这两个求解器。')
    parser.add_argument('--mu', default=1e-2, type=float, help='正则项的系数mu。默认为0.01。')
    parser.add_argument('--seed', '-RS', default=97108120, type=int, help='指定测试数据的随机数种子。默认为97108120，为 `alx` 的ASCII码依次排列。') # seed = ord("a") ord("l") ord("x")
    parser.add_argument('--plot', '-P', action='store_true', help='表明是否绘制迭代曲线，如果增加此参数，则绘制。')
    args = parser.parse_args()

    if len(args.solvers) == 1 and str.lower(args.solvers[0]) == 'all':
        args.solvers = [
            'gl_cvx_gurobi',
            'gl_cvx_mosek',
            'gl_gurobi',
            'gl_mosek',
            # 'gl_ADMM_dual',
            # 'gl_ADMM_primal_direct', 
            # 'gl_ADMM_primal',
            # 'gl_FGD_primal', 
            # 'gl_FGD_primal_line_search',
            # 'gl_ProxGD_primal', 
            # 'gl_ProxGD_primal_line_search',
            # 'gl_FProxGD_primal', 
            # 'gl_FProxGD_primal_line_search',
            # 'gl_SGD_primal', 
            # 'gl_SGD_primal_normal_sgd',
            # 'gl_GD_primal', 
            # 'gl_GD_primal_normal_gd',
            # 'gl_gurobi_term',
        ]

    # 初始化测试数据
    x0, A, b, mu, u, f_u, sparsity_u = utils.testData(seed=args.seed, mu=args.mu)
    utils.logger.info(f"Solver: {args.solvers}")
    utils.logger.info(f"mu: {args.mu}")
    utils.logger.info(f"random seed: {args.seed}")
    utils.logger.info(f"is_plot: {args.plot}")
    tab = []
    with open(utils.cvxLogsName, "w", encoding='utf-8') as devlog, utils.RedirectStdStreams(stdout=devlog, stderr=devlog):
        for solver_name in args.solvers:
            utils.logger.info(f"\n--->Current Test Solver: {solver_name}<---")
            solver = getattr(importlib.import_module("code." + solver_name), solver_name)
            tic = time.time()
            x, iters_N, out = solver(x0.copy(), A, b, mu) 
            toc = time.time()
            time_cpu = toc - tic
            utils.logger.info(f"Current Solver takes {time_cpu}, with {iters_N} iters")

            obj = out['obj']
            iters = out['iters']
            err_x_u = utils.errFun(x, u)
            sparsity_x = utils.sparsity(x)
            if iters_N == 0:
                utils.logger.error(f"记录迭代次数为0，跳过该求解器")
                continue
            x, y = zip(*iters)
            if y[0]<0: # 使用mosek求解时，会出现y[0]为负数的情况，这里将其去除
                x = x[1:]
                y = y[1:]
            if args.plot:
                plt.plot(x, y, '*-', label=(solver_name[3:] + " in " + str(iters_N) + " iters"))
                utils.logger.info(f"Plot curve for {solver_name}")
            tab.append([solver_name[3:], obj, err_x_u, time_cpu, iters_N, sparsity_x])
            utils.cleanUpLog()

    utils.logger.info(f"\n#######==ALL solvers have finished==#######")
    utils.logger.info(f"问题精确解的目标函数值f_u: {f_u}")
    utils.logger.info(f"问题精确解的稀疏度sparsity_u: {sparsity_u}")
    utils.logger.info("\n"+tabulate(tab, headers=['Solver', 'Objective', 'x_u_Error', 'Time(s)', 'Iter', 'Sparsity']))
    if args.plot:
        plt.yscale('log')
        plt.legend()
        plt.show()


        