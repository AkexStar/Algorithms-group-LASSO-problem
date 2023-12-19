import src.utils as utils
import time
import importlib
import argparse
import copy
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

def testData(opts):
    utils.logger.info(f"testData opts: {opts}")
    seed = int(opts.get("seed", 97108120)) # seed = ord("a") ord("l") ord("x")
    mu = float(opts.get("mu", 1e-2))
    n = int(opts.get("n", 512))
    m = int(opts.get("m", 256))
    l = int(opts.get("l", 2))
    r = float(opts.get("r", 1e-1))
    np.random.seed(seed)
    A = np.random.randn(m, n)
    k = round(n * r)
    p = np.random.permutation(n)[:k]
    u = np.zeros((n, l))
    u[p, :] = np.random.randn(k, l)
    b = A @ u
    # x0 = u + np.random.rand(n, l) * 0.001
    # x0 = np.zeros((n, l))
    x0 = np.random.randn(n, l)
    f_u = utils.objFun(u, A, b, mu)
    sparsity_u = utils.sparsity(u)
    return x0, A, b, mu, u, f_u, sparsity_u

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Test_group_lasso', description='测试不同的求解器进行group lasso求解')
    # parser.add_argument('--solvers', '-S', nargs='+', default=['gl_cvx_gurobi', 'gl_cvx_mosek'], help='指定求解器名称, 输入`all` 可以测试本项目中所有的求解器函数。默认填充为 `[\'gl_cvx_gurobi\', \'gl_cvx_mosek\']` 这两个求解器。')
    parser.add_argument('--solvers', '-S', nargs='+', default=['all'], help='指定求解器名称, 输入`all` 可以测试本项目中所有的求解器函数。默认填充为 `[\'gl_cvx_gurobi\', \'gl_cvx_mosek\']` 这两个求解器。')
    # parser.add_argument('--mu', default=1e-2, type=float, help='正则项的系数mu。默认为0.01。')
    parser.add_argument('--seed', '-RS', default=97108120, type=int, help='指定测试数据的随机数种子。默认为97108120，为 `alx` 的ASCII码依次排列。') # seed = ord("a") ord("l") ord("x")
    parser.add_argument('--plot', '-P', action='store_true', help='表明是否绘制迭代曲线，如果增加此参数，则绘制。')
    parser.add_argument('--log', '-L', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='指定日志等级。默认为INFO。')
    parser.add_argument('--opts', '-O', nargs='+', default={}, type=lambda kv: kv.split("="), help='指定测试数据的参数，格式为`key=value`，可以有多个。默认为空。')
    args = parser.parse_args()
    utils.logger.debug(f"raw opts: {args.opts}")
    if any(len(kv) < 2 for kv in args.opts): 
        raise ValueError('opts参数字典必须以此形式给出 "KEY=VALUE"')
    # from ast import literal_eval # 用于将字符串转换为字典
    # d = literal_eval(p)
    opts = dict(args.opts)
    if len(args.solvers) == 1 and str.lower(args.solvers[0]) == 'all':
        args.solvers = utils.solversCollection

    # 打印输入参数
    utils.logger.info(f"test solvers: {args.solvers}")
    utils.logger.info(f"opts: {opts}")
    utils.logger.info(f"random seed: {args.seed}")
    utils.logger.info(f"is plot?: {args.plot}")
    utils.logger.info(f"log level: {args.log}")
    utils.logger.setLevel(args.log)

    # 初始化测试数据
    x0, A, b, mu, u, f_u, sparsity_u = testData(opts)
    utils.logger.info(f"问题精确解的目标函数值f_u: {f_u}")
    utils.logger.info(f"问题精确解的稀疏度sparsity_u: {sparsity_u}")

    # 测试结果表格
    tab = []

    # 重定向输出流
    with open(utils.cvxLogsName, "w", encoding='utf-8') as devlog, utils.RedirectStdStreams(stdout=devlog, stderr=devlog):
        # 遍历测试每个求解器
        for solver_name in args.solvers:
            utils.logger.info(f"\n--->Current Test Solver: {solver_name}<---")
            solver = getattr(importlib.import_module("src." + solver_name), solver_name)
            tic = time.time()
            x, iters_N, out = solver(copy.deepcopy(x0), A, b, mu, opts) 
            toc = time.time()
            time_cpu = toc - tic
            utils.logger.info(f"Current Solver takes {time_cpu:.5f}s, with {iters_N} iters")

            fval = out['fval']
            iters = out['iters']
            utils.logger.debug(f"iters: {iters}")
            utils.logger.info(f"fval: {fval}")
            utils.logger.debug(f"x.shape: {x.shape}")
            err_x_u = utils.errX(x, u)
            sparsity_x = utils.sparsity(x)
            if iters_N == 0:
                utils.logger.error(f"求解器{solver_name}的记录迭代次数为0，跳过该求解器。需要检查日志文件{utils.cvxLogsName}")
                continue
            x, y = zip(*iters)
            utils.logger.debug(f"len(x)={len(x)}")
            utils.logger.debug(f"len(y)={len(y)}")
            if y[0]<0: # 使用mosek求解时，会出现y[0]为负数的情况，这里将其去除
                x = x[1:]; y = y[1:]
            if args.plot:
                plt.plot(x, y, '.-', label=(solver_name[3:] + " in " + str(iters_N) + " iters"))
                utils.logger.info(f"Plot curve for {solver_name}")
            tab.append([solver_name[3:], fval, utils.errObj(fval, f_u), err_x_u, time_cpu, iters_N, sparsity_x])
            utils.cleanUpLog()

    utils.logger.info(f"\n#######==ALL solvers have finished==#######")
    utils.logger.info(f"问题精确解的目标函数值f_u: {f_u}")
    utils.logger.info(f"问题精确解的稀疏度sparsity_u: {sparsity_u}")
    utils.logger.info("\n"+tabulate(tab, headers=['Solver', 'Objective', 'Obj_ABS_Error', 'x_u_Error', 'Time(s)', 'Iter', 'Sparsity']))
    if args.plot:
        plt.yscale('log')
        plt.legend()
        plt.show()


        