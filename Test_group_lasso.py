import src.utils as utils
import time
import importlib
import argparse
import copy
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

# 生成测试数据
def testData(opts:dict = {}):
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
    # sparsity_u = utils.sparsity(u)
    return x0, A, b, mu, u, f_u

# 与cvx调用的mosek和gurobi比较
def compareXWith(x0, A, b, mu, u, f_u, opts:dict = {}):
    solver_name_mosek = 'gl_cvx_mosek'
    solver_mosek = getattr(importlib.import_module("src." + solver_name_mosek), solver_name_mosek)
    solver_name_gurobi = 'gl_cvx_gurobi'
    solver_gurobi = getattr(importlib.import_module("src." + solver_name_gurobi), solver_name_gurobi)
    tab = []
    
    tic = time.time()
    x_mosek, iters_N_mosek, out_mosek = solver_mosek(copy.deepcopy(x0), A, b, mu, opts) 
    toc = time.time()
    time_cpu_mosek = toc - tic
    utils.cleanUpLog()
    fval_mosek = out_mosek['fval']
    err_x_u_mosek = utils.errX(x_mosek, u)
    sparsity_x_mosek = utils.sparsity(x_mosek)
    utils.logger.debug(f"iters_N_mosek: {iters_N_mosek}")
    utils.logger.debug(f"fval_mosek: {fval_mosek}")
    utils.logger.debug(f"out_mosek['iters']: {out_mosek['iters']}")
    
    tic = time.time()
    x_gurobi, iters_N_gurobi, out_gurobi = solver_gurobi(copy.deepcopy(x0), A, b, mu, opts)
    toc = time.time()
    time_cpu_gurobi = toc - tic
    utils.cleanUpLog()
    fval_gurobi = out_gurobi['fval']
    err_x_u_gurobi = utils.errX(x_gurobi, u)
    sparsity_x_gurobi = utils.sparsity(x_gurobi)
    utils.logger.debug(f"iters_N_gurobi: {iters_N_gurobi}")
    utils.logger.debug(f"fval_gurobi: {fval_gurobi}")
    utils.logger.debug(f"out_gurobi['iters']: {out_gurobi['iters']}]")

    tab.append([
        solver_name_mosek[3:], fval_mosek, utils.errObj(fval_mosek, f_u), 
        err_x_u_mosek, utils.errX(x_mosek, x_mosek), utils.errX(x_mosek, x_gurobi), 
        time_cpu_mosek, iters_N_mosek, sparsity_x_mosek])

    tab.append([
        solver_name_gurobi[3:], fval_gurobi, utils.errObj(fval_gurobi, f_u), 
        err_x_u_gurobi, utils.errX(x_gurobi, x_mosek), utils.errX(x_gurobi, x_gurobi),
        time_cpu_gurobi, iters_N_gurobi, sparsity_x_gurobi])
        
    return tab, x_mosek, x_gurobi

if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser(prog='Test_group_lasso', description='测试不同的求解器进行group-lasso求解')
    # parser.add_argument('--solvers', '-S', nargs='+', default=['gl_cvx_gurobi', 'gl_cvx_mosek'], help='指定求解器名称, 输入`all` 可以测试本项目中所有的求解器函数。默认填充为 `[\'gl_cvx_gurobi\', \'gl_cvx_mosek\']` 这两个求解器。')
    parser.add_argument('--solvers', '-S', nargs='+', default=['all'], help='指定求解器名称, 输入`all` 可以测试本项目中所有的求解器函数。默认填充为 `[\'gl_cvx_gurobi\', \'gl_cvx_mosek\']` 这两个求解器。')
    parser.add_argument('--seed', '-RS', default=97108120, type=int, help='指定测试数据的随机数种子。默认为97108120，为 `alx` 的ASCII码依次排列。') # seed = ord("a") ord("l") ord("x")
    parser.add_argument('--plot', '-P', action='store_true', help='表明是否绘制迭代曲线，如果增加此参数，则绘制。')
    parser.add_argument('--log', '-L', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='指定日志等级。默认为INFO。')
    parser.add_argument('--opts', '-O', nargs='+', default={}, type=lambda kv: kv.split("="), help='指定测试数据的参数，格式为`key=value`，可以有多个。默认为空。')
    parser.add_argument('--compare', '-C', action='store_true', help='表明是否将计算得到的最优解与mosek和gurobi的结果比较，如果增加此参数，则比较。')
    args = parser.parse_args()
    utils.logger.setLevel(args.log)
    # 处理opts参数
    utils.logger.debug(f"raw opts: {args.opts}")
    if any(len(kv) < 2 for kv in args.opts): 
        raise ValueError('opts参数字典必须以此形式给出 "KEY=VALUE"')
    # from ast import literal_eval # 用于将字符串转换为字典
    # d = literal_eval(p)
    opts = dict(args.opts)
    # 处理solvers参数，如果是all，则替换为所有求解器
    if len(args.solvers) == 1 and str.lower(args.solvers[0]) == 'all':
        args.solvers = utils.solversCollection

    # 打印输入参数
    utils.logger.info(f"test solvers: {args.solvers}")
    utils.logger.info(f"opts: {opts}")
    utils.logger.info(f"random seed: {args.seed}")
    utils.logger.info(f"is plot?: {args.plot}")
    utils.logger.info(f"is compare?: {args.compare}")
    utils.logger.info(f"log level: {args.log}")

    # 初始化测试数据
    x0, A, b, mu, u, f_u = testData(dict(opts.get('testData', {})))
    sparsity_u = utils.sparsity(u)
    utils.logger.info(f"问题精确解的目标函数值f_u: {f_u}")
    utils.logger.info(f"问题精确解的稀疏度sparsity_u: {sparsity_u}")
    utils.logger.info(f"\n#######==Start all solvers TEST==#######")

    # 测试结果表格
    tab = []

    # 处理compare参数
    if args.compare:
        with open(utils.cvxLogsName, "w", encoding='utf-8') as devlog, utils.RedirectStdStreams(stdout=devlog, stderr=devlog):
            tab, x_mosek, x_gurobi = compareXWith(x0, A, b, mu, u, f_u, opts)

    # 重定向输出流
    with open(utils.cvxLogsName, "w", encoding='utf-8') as devlog, utils.RedirectStdStreams(stdout=devlog, stderr=devlog):
        # 遍历测试每个求解器
        for solver_name in args.solvers:
            # if solver_name not in utils.solversCollection:
            #     utils.logger.error(f"求解器{solver_name}不存在，跳过该求解器。")
            #     continue
            if args.compare and solver_name in ['gl_cvx_gurobi', 'gl_cvx_mosek']:
                utils.logger.info(f"求解器{solver_name}已经在compare中测试，跳过该求解器。")
                continue
            # 分别调用每个求解器函数 脚本名称和函数名称一致
            utils.logger.info(f"\n--->Current Test Solver: {solver_name}<---")
            try:
                solver = getattr(importlib.import_module("src." + solver_name), solver_name)
            except AttributeError:
                utils.logger.error(f"求解器{solver_name}不存在，跳过该求解器。")
                continue
            # 取出求解器的参数
            solver_opts = dict(opts.get(solver_name[3:], {}))
            utils.logger.info(f"solver_opts: {solver_opts}")
            # 测试求解器并记录时间
            tic = time.time()
            x, iters_N, out = solver(copy.deepcopy(x0), A, b, mu, solver_opts) 
            toc = time.time()
            time_cpu = toc - tic
            utils.cleanUpLog()
            utils.logger.info(f"Current Solver takes {time_cpu:.5f}s, with {iters_N} iters")
            # 处理求解器的输出
            fval = out['fval']
            iters = out['iters']
            utils.logger.debug(f"每次迭代情况 iters:\n{iters}")
            utils.logger.info(f"最终目标函数值 fval: {fval}")
            err_x_u = utils.errX(x, u) # 计算x的误差
            sparsity_x = utils.sparsity(x) # 计算x的稀疏度
            if iters_N == 0 or iters_N == -1:
                utils.logger.error(f"{solver_name}'s iters_N = {iters_N}，跳过该求解器！且需要检查stdout重定向情况和日志文件{utils.cvxLogsName}")
                continue
            # 绘制迭代曲线
            plot_x, plot_y = zip(*iters)
            utils.logger.debug(f"len(x)={len(plot_x)}")
            utils.logger.debug(f"len(y)={len(plot_y)}")
            if plot_y[0]<0: # 使用mosek求解时，会出现y[0]为负数的情况，这里将其去除
                plot_x = plot_x[1:]; plot_y = plot_y[1:]
            if args.plot:
                plt.plot(plot_x, plot_y, '.-', label=(solver_name[3:] + " in " + str(iters_N) + " iters"))
                utils.logger.info(f"Plot curve for {solver_name}")
            # 制作结果比较表格
            if args.compare:
                tab.append([solver_name[3:], fval, utils.errObj(fval, f_u), 
                            err_x_u, utils.errX(x, x_mosek), utils.errX(x, x_gurobi), 
                            time_cpu, iters_N, sparsity_x])
            else:
                tab.append([solver_name[3:], fval, utils.errObj(fval, f_u), err_x_u, time_cpu, iters_N, sparsity_x])
            

    utils.logger.info(f"\n#######==ALL solvers have finished==#######")
    utils.logger.info(f"问题精确解的目标函数值f_u: {f_u}")
    utils.logger.info(f"问题精确解的稀疏度sparsity_u: {sparsity_u}")
    if args.compare:
        # utils.logger.info(f"使用mosek求解得到的最优解x_mosek: {x_mosek}")
        # utils.logger.info(f"使用gurobi求解得到的最优解x_gurobi: {x_gurobi}")
        tabulate_headers = ['Solver', 'Objective', 'Obj_ABS_Error', 'x_u_Error', 'x_CVXmosek_Error', 'x_CVXgurobi_Error', 'Time(s)', 'Iter', 'Sparsity']
    else:
        tabulate_headers = ['Solver', 'Objective', 'Obj_ABS_Error', 'x_u_Error', 'Time(s)', 'Iter', 'Sparsity']
    utils.logger.info("\n"+tabulate(tab, headers=tabulate_headers))
    print("\n")
    print(tabulate(tab, headers=tabulate_headers))
    if args.plot:
        plt.yscale('log')
        plt.legend()
        plt.show()


        