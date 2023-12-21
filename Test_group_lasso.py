import src.utils as utils
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser(prog='Test_group_lasso', description='测试不同的求解器进行group-lasso求解 https://github.com/AkexStar/Algorithms-group-LASSO-problem')
    parser.add_argument('--solvers', '-S', nargs='+', default=['gl_cvx_gurobi', 'gl_cvx_mosek'], help='指定求解器名称, 输入`all` 可以测试本项目中所有的求解器函数。默认填充为 `[\'gl_cvx_gurobi\', \'gl_cvx_mosek\']` 这两个求解器。')
    # parser.add_argument('--solvers', '-S', nargs='+', default=['all'], help='指定求解器名称, 输入`all` 可以测试本项目中所有的求解器函数。默认填充为 `[\'gl_cvx_gurobi\', \'gl_cvx_mosek\']` 这两个求解器。')
    parser.add_argument('--seed', '-RS', default=97108120, type=int, help='指定测试数据的随机数种子。默认为97108120，为 `alx` 的ASCII码依次排列。') # seed = ord("a") ord("l") ord("x")
    parser.add_argument('--plot', '-P', action='store_true', help='表明是否绘制迭代曲线，如果增加此参数，则绘制。')
    parser.add_argument('--log', '-L', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='指定日志等级。默认为INFO。')
    parser.add_argument('--opts', '-O', nargs='+', default={}, type=lambda kv: kv.split("="), help='指定测试数据的参数，格式为`key=value`，可以有多个。例如 `-O gl_ALM_dual={\'maxit\':60, \'maxit_inn\':100} testData={\'m\'=256, \'n\':512}` 。没有指定的参数将使用默认值。')
    parser.add_argument('--compare', '-C', action='store_true', help='表明是否将计算得到的最优解与mosek和gurobi的结果比较，如果增加此参数，则比较。')
    parser.add_argument('--version', '-V', action='version', version='%(prog)s 1.0 2023-12-21')
    parser.add_argument('--printDefaultOpts', '-PDO', action='store_true', help='展示所有默认opts参数。')
    args = parser.parse_args()
    utils.logger.setLevel(args.log)
    if args.printDefaultOpts:
        utils.printAllDefaultOpts()
        exit(0)
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
    x0, A, b, mu, u, f_u = utils.testData(dict(opts.get('testData', {})))
    sparsity_u = utils.sparsity(u)
    utils.logger.debug(f"问题精确解的目标函数值f_u: {f_u}")
    utils.logger.debug(f"问题精确解的稀疏度sparsity_u: {sparsity_u}")
    utils.logger.info(f"\n#######==Start all solvers TEST==#######")

    # 测试结果表格
    tab = []

    # 重定向输出流
    with open(utils.cvxLogsName, "w", encoding='utf-8') as devlog, utils.RedirectStdStreams(stdout=devlog, stderr=devlog):
        # 处理compare参数
        if args.compare:
            utils.logger.info(f"\n--->Compare Solver: cvx_mosek and cvx_gurobi<---")
            x_mosek, iters_N_mosek, out_mosek = utils.testSolver(x0, A, b, mu, {'solver_name': 'gl_cvx_mosek'})
            x_gurobi, iters_N_gurobi, out_gurobi = utils.testSolver(x0, A, b, mu, {'solver_name': 'gl_cvx_gurobi'})
            tab.append([
                'cvx_mosek', out_mosek['fval'], utils.errObj(out_mosek['fval'], f_u), 
                utils.errX(x_mosek, u), utils.errX(x_mosek, x_mosek), utils.errX(x_mosek, x_gurobi), 
                out_mosek['time_cpu'], iters_N_mosek, out_mosek['sparsity_x']])
            tab.append([
                'cvx_gurobi', out_gurobi['fval'], utils.errObj(out_gurobi['fval'], f_u), 
                utils.errX(x_gurobi, u), utils.errX(x_gurobi, x_mosek), utils.errX(x_gurobi, x_gurobi),
                out_gurobi['time_cpu'], iters_N_gurobi, out_gurobi['sparsity_x']])
            if args.plot:
                plot_x_mosek, plot_y_mosek = zip(*out_mosek['iters'])
                plot_x_gurobi, plot_y_gurobi = zip(*out_gurobi['iters'])
                plt.plot(plot_x_mosek, plot_y_mosek, '.-', label=('cvx_mosek in ' + str(iters_N_mosek) + ' iters'))
                plt.plot(plot_x_gurobi, plot_y_gurobi, '.-', label=('cvx_gurobi in ' + str(iters_N_gurobi) + ' iters'))
                utils.logger.info(f"Plot curve for cvx_mosek and cvx_gurobi")
        # 遍历测试每个求解器
        for solver_name in args.solvers:
            # 检查求解器是否存在
            if solver_name not in utils.solversCollection:
                utils.logger.error(f"求解器{solver_name}不存在，跳过该求解器。")
                continue
            # 检查求解器是否已经在compare中测试
            if args.compare and solver_name in ['gl_cvx_gurobi', 'gl_cvx_mosek']:
                utils.logger.info(f"求解器{solver_name}已经在compare中测试，跳过该求解器。")
                continue
            # 测试求解器
            solver_opts = dict(opts.get(solver_name, {}))
            solver_opts['solver_name'] = solver_name
            x, iters_N, out = utils.testSolver(x0, A, b, mu, solver_opts)            
            # 处理求解器的输出
            err_x_u = utils.errX(x, u) # 计算x的误差
            sparsity_x = utils.sparsity(x) # 计算x的稀疏度
            if iters_N == 0 or iters_N == -1:
                utils.logger.error(f"{solver_name}'s iters_N = {iters_N}，跳过该求解器！且需要检查stdout重定向情况和日志文件{utils.cvxLogsName}")
                continue
            # 绘制迭代曲线
            plot_x, plot_y = zip(*out['iters'])
            utils.logger.debug(f"len(x)={len(plot_x)}")
            utils.logger.debug(f"len(y)={len(plot_y)}")
            if plot_y[0]<0: # 使用mosek求解时，会出现y[0]为负数的情况，这里将其去除
                plot_x = plot_x[1:]; plot_y = plot_y[1:]
            if args.plot:
                plt.plot(plot_x, plot_y, '.-', label=(solver_name[3:] + " in " + str(iters_N) + " iters"))
                utils.logger.info(f"Plot curve for {solver_name}")
            # 制作结果比较表格
            if args.compare:
                tab.append([solver_name[3:], out['fval'], utils.errObj(out['fval'], f_u), 
                            err_x_u, utils.errX(x, x_mosek), utils.errX(x, x_gurobi), 
                            out['time_cpu'], iters_N, sparsity_x])
            else:
                tab.append([solver_name[3:], out['fval'], utils.errObj(out['fval'], f_u), err_x_u, out['time_cpu'], iters_N, sparsity_x])

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


        