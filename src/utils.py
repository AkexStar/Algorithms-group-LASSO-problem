import sys
import os
import re
import time
import logging
import time
import importlib
import numpy as np


def setLoggerLevel(logger, level: str):
    """设置日志级别
    Args:
        - level (str): 日志级别，可选值为DEBUG, INFO, WARNING, ERROR, CRITICAL，默认为INFO
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('输入不正确的日志级别: %s' % level)
    logger.setLevel(numeric_level)
    logger.info(f"日志级别设置为{level.upper()}")

# 初始化日志对象
def loggerInit(name: str = None):
    # 使用一个名字为ATC-CDG的logger
    logger = logging.getLogger(name)
    # 设置logger的level为INFO
    setLoggerLevel(logger, 'INFO')

    # 创建一个输出日志到控制台的StreamHandler
    hdr = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    hdr.setFormatter(formatter)
    # 给logger添加上handler
    logger.addHandler(hdr)

    # 同时写入日志文件
    current_work_dir = os.path.dirname(__file__)
    if not os.path.exists(current_work_dir+'/logs'):
        os.makedirs(current_work_dir+'/logs')
    now = time.strftime("%m%d-%H%M%S",time.localtime(time.time()))
    loggerName = current_work_dir+f'/logs/{name}-{now}.log'
    cvxLogsName = current_work_dir+f'/logs/gl_cvx.log'
    logging.basicConfig(filename = loggerName,
                        level = logging.INFO,
                        encoding='utf-8',
                        format = '[%(asctime)s] %(filename)s: %(funcName)s: %(levelname)s: %(message)s')
    logger.debug(f"日志文件保存在: {current_work_dir}\logs\{name}{now}.log")
    return logger, loggerName, cvxLogsName

logger, loggerName, cvxLogsName = loggerInit('AGLP')

# 重定向stdout
class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

re_iterc_default = re.compile(r'^ *(?P<iterc>\d{1,3})\:? +(?P<objv>[0-9\.eE\+\-]+)', re.MULTILINE)

# 正则表达式字典
reg_solver = {
    'GUROBI': re_iterc_default,
    # ([\s\S]{26})\:( +)(\d{1,2}) ([\s\S]{38})( +)([\-\+0-9\.eE]+)
    # 正则表达式赛高 排除前26个字符，匹配冒号，匹配空格，匹配1-2个数字，排除前38个字符，匹配空格，匹配数字，匹配小数点，匹配e，匹配E，匹配+，匹配-，匹配数字
    'MOSEK': re.compile(r'^ *([\s\S]{26})\:( +)(?P<iterc>\d{1,2}) ([\s\S]{38})( +)(?P<objv>[\-\+0-9\.eE]+)', re.MULTILINE), 
    'MOSEK_OLD': re.compile(r'^ *(?P<iterc>\d{1,3})\:?( +(?:[0-9\.eE\+\-]+)){4} +(?P<objv>[0-9\.eE\+\-]+)', re.MULTILINE),   # skip four columns
    'CVXOPT': re_iterc_default,
}

# 解析CVX输出日志文件
def parse_iters(s, solver=None):

    re_iterc = reg_solver[solver] if solver in reg_solver else re_iterc_default
    ret = []
    for match in re_iterc.finditer(s):
        ret.append((int(match.groupdict()['iterc']),
                    float(match.groupdict()['objv'])))
    return ret

# 清空CVX输出日志文件
def cleanUpLog():
    with open(cvxLogsName,'w+',encoding='utf-8') as test:
        test.truncate(0)
        for line in test.readlines():
            line.replace(r'\0', '')

# 计算稀疏度
def sparsity(x) -> float:
    # return np.sum(np.abs(x) > 1e-6 * np.max(np.abs(x))) / x.size
    return np.sum(np.abs(x) > 1e-5) / x.size
    # return np.sum(x <= 1e-5) / np.sum(np.ones_like(x))

# 计算解之间的区别
def errX(x, x0) -> float:
    # return np.linalg.norm(x - u, ord='fro')
    # return np.linalg.norm(u - x, 'fro') / np.linalg.norm(u)
    return np.linalg.norm(x - x0, 'fro') / (1 + np.linalg.norm(x0, 'fro'))

# 计算目标函数值
def objFun(x, A, b, mu) -> float:
    # r = np.dot(A, x) - b
    # return 0.5 * np.linalg.norm(r, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
    return 0.5 * np.linalg.norm(A @ x - b, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))

# 计算目标函数值之间差的绝对值
def errObj(obj, obj0) -> float:
    return np.abs(obj - obj0)

# 邻近算子
def prox(x, mu):
    nrmx = np.linalg.norm(x, ord=2, axis=1).reshape(-1, 1)
    flag = nrmx > mu
    prox_x = x - mu * x  / (nrmx + 1e-10)
    prox_x = prox_x * flag
    return prox_x

# BB步长更新
def BBupdate(x, xp, g, gp, k, alpha):
    dx = x - xp
    dg = g - gp
    dxg = np.abs(np.sum(dx * dg))
    # if dxg > 0:
    if dxg > 1e-12:
        if np.mod(k, 2) == 1:
            alpha = (np.sum(dx * dx) / dxg)
        else:
            alpha = (dxg / np.sum(dg * dg))
    return max(min(alpha, 1e12), 1e-12)

def testDataParams(opts0:dict = {}):
    opts = {}
    opts['seed'] = int(opts0.get("seed", 97108120)) # seed = ord("a") ord("l") ord("x")
    opts['mu'] = float(opts0.get("mu", 1e-2))
    opts['n'] = int(opts0.get("n", 512))
    opts['m'] = int(opts0.get("m", 256))
    opts['l'] = int(opts0.get("l", 2))
    opts['r'] = float(opts0.get("r", 1e-1))
    return opts

# 生成测试数据
def testData(opts:dict = {}):
    opts = testDataParams(opts)
    logger.info(f"testData opts: {opts}")
    np.random.seed(opts['seed'])
    m = opts['m']
    n = opts['n']
    l = opts['l']
    r = opts['r']
    mu = opts['mu']
    A = np.random.randn(m, n)
    k = round(n * r)
    p = np.random.permutation(n)[:k]
    u = np.zeros((n, l))
    u[p, :] = np.random.randn(k, l)
    b = A @ u
    # x0 = u + np.random.rand(n, l) * 0.001
    # x0 = np.zeros((n, l))
    x0 = np.random.randn(n, l)
    f_u = objFun(u, A, b, mu)
    # sparsity_u = utils.sparsity(u)
    return x0, A, b, mu, u, f_u

def testSolver(x0, A, b, mu, opts:dict = {}):
    # 获得求解器名称
    solver_name = opts.get('solver_name', '')
    if solver_name == '':
        raise ValueError('opts参数字典中必须包含solver_name键值对，指定求解器名称')
    # 检查求解器是否存在
    try:
        solver = getattr(importlib.import_module("src." + solver_name), solver_name)
    except AttributeError:
        logger.error(f"求解器{solver_name}不存在，跳过该求解器。")
        return None, None, None
    logger.info(f"\n--->Current Test Solver: {solver_name}<---")
    # 取出求解器的参数
    solver_opts = dict(opts.get(solver_name[3:], {}))
    logger.info(f"solver_opts: {solver_opts}")
    # 测试求解器并记录时间
    tic = time.time()
    x, iters_N, out = solver(x0, A, b, mu, solver_opts) 
    toc = time.time()
    time_cpu = toc - tic
    cleanUpLog()
    out['time_cpu'] = time_cpu
    sparsity_x = sparsity(x)
    out['sparsity_x'] = sparsity_x
    logger.info(f"{solver_name[3:]} takes {time_cpu:.5f}s, with {iters_N} iters")
    logger.debug(f"out['fval']: {out['fval']}")
    logger.debug(f"sparsity_x: {sparsity_x}")
    logger.debug(f"out['iters']: \n{out['iters']}")
    return x, iters_N, out

# 项目已经实现的所有求解器
solversCollection = [
    'gl_cvx_mosek',
    'gl_cvx_gurobi',
    'gl_mosek',
    'gl_gurobi',
    
    'gl_SGD_primal',
    'gl_ProxGD_primal',
    'gl_FProxGD_primal', 
    'gl_ALM_dual',
    'gl_ADMM_dual',
    'gl_ADMM_primal',
    # 'gl_FGD_primal', 
    # 'gl_GD_primal', 
]

# 次梯度法默认参数
def SGD_primal_optsInit(opts0: dict = {}):
    opts = {}
    opts['maxit'] = int(opts0.get('maxit', 50)) # 连续化策略最大迭代次数
    opts['maxit_inn'] = int(opts0.get('maxit_inn', 250)) # 内循环最大迭代次数

    opts['ftol'] = float(opts0.get('ftol', 1e-9)) # 针对函数值的停机判断条件
    opts['ftol_init_ratio'] = float(opts0.get('ftol_init_ratio', 1e6)) # 初始时停机准则 opts['ftol'] 的放大倍数
    opts['etaf'] = float(opts0.get('etaf', 0.1)) # 每步外层循环的停机判断标准 opts['ftol'] 的缩减倍率

    opts['gtol'] = float(opts0.get('gtol', 1e-6)) # 针对梯度的停机判断条件
    opts['gtol_init_ratio'] = float(opts0.get('gtol_init_ratio', 1 / opts['gtol'])) # 初始时停机准则 opts['gtol'] 的放大倍数
    opts['etag'] = float(opts0.get('etag', 0.1)) # 每步外层循环的停机判断标准 opts['gtol'] 的缩减倍率

    opts['factor'] = float(opts0.get('factor', 0.1)) # 正则化系数的衰减率
    opts['mu1'] = float(opts0.get('mu1', 10)) # 初始的正则化系数（采用连续化策略，从更大的正则化系数开始）
    
    opts['is_only_print_outer'] = bool(opts0.get('is_only_print_outer', False)) # 是否只打印外循环的信息
    opts['method'] = opts0.get('method', None) # 内循环使用的求解器

    # 针对内循环的参数
    opts['gamma'] = float(opts0.get('gamma', 0.9)) 
    opts['rhols'] = float(opts0.get('rhols', 1e-6)) # 线搜索的参数
    opts['eta'] = float(opts0.get('eta', 0.2)) # 线搜索的参数
    opts['Q'] = float(opts0.get('Q', 1)) # 线搜索的参数
    return opts

# 近似点梯度法默认参数
def ProxGD_primal_optsInit(opts0: dict = {}):
    opts = {}
    opts['maxit'] = int(opts0.get('maxit', 50)) # 连续化策略最大迭代次数
    opts['maxit_inn'] = int(opts0.get('maxit_inn', 250)) # 内循环最大迭代次数

    opts['ftol'] = float(opts0.get('ftol', 1e-9)) # 针对函数值的停机判断条件
    opts['ftol_init_ratio'] = float(opts0.get('ftol_init_ratio', 1e6)) # 初始时停机准则 opts['ftol'] 的放大倍数
    opts['etaf'] = float(opts0.get('etaf', 0.1)) # 每步外层循环的停机判断标准 opts['ftol'] 的缩减倍率

    opts['gtol'] = float(opts0.get('gtol', 1e-6)) # 针对梯度的停机判断条件
    opts['gtol_init_ratio'] = float(opts0.get('gtol_init_ratio', 1 / opts['gtol'])) # 初始时停机准则 opts['gtol'] 的放大倍数
    opts['etag'] = float(opts0.get('etag', 0.1)) # 每步外层循环的停机判断标准 opts['gtol'] 的缩减倍率

    opts['factor'] = float(opts0.get('factor', 0.1)) # 正则化系数的衰减率
    opts['mu1'] = float(opts0.get('mu1', 10)) # 初始的正则化系数（采用连续化策略，从更大的正则化系数开始）
    
    opts['is_only_print_outer'] = bool(opts0.get('is_only_print_outer', False)) # 是否只打印外循环的信息
    opts['method'] = opts0.get('method', None) # 内循环使用的求解器

    # 针对内循环的参数
    opts['gamma'] = float(opts0.get('gamma', 0.85))
    opts['rhols'] = float(opts0.get('rhols', 1e-6)) # 线搜索的参数
    opts['eta'] = float(opts0.get('eta', 0.2)) # 线搜索的参数
    opts['Q'] = float(opts0.get('Q', 1)) # 线搜索的参数

    return opts

# 快速近似梯度法默认参数
def FProxGD_primal_optsInit(opts0: dict = {}):
    opts = {}
    opts['maxit'] = int(opts0.get('maxit', 50)) # 连续化策略最大迭代次数
    opts['maxit_inn'] = int(opts0.get('maxit_inn', 250)) # 内循环最大迭代次数

    opts['ftol'] = float(opts0.get('ftol', 1e-9)) # 针对函数值的停机判断条件
    opts['ftol_init_ratio'] = float(opts0.get('ftol_init_ratio', 1e6)) # 初始时停机准则 opts['ftol'] 的放大倍数
    opts['etaf'] = float(opts0.get('etaf', 0.1)) # 每步外层循环的停机判断标准 opts['ftol'] 的缩减倍率

    opts['gtol'] = float(opts0.get('gtol', 1e-6)) # 针对梯度的停机判断条件
    opts['gtol_init_ratio'] = float(opts0.get('gtol_init_ratio', 1 / opts['gtol'])) # 初始时停机准则 opts['gtol'] 的放大倍数
    opts['etag'] = float(opts0.get('etag', 0.1)) # 每步外层循环的停机判断标准 opts['gtol'] 的缩减倍率

    opts['factor'] = float(opts0.get('factor', 0.1)) # 正则化系数的衰减率
    opts['mu1'] = float(opts0.get('mu1', 10)) # 初始的正则化系数（采用连续化策略，从更大的正则化系数开始）
    
    opts['is_only_print_outer'] = bool(opts0.get('is_only_print_outer', False)) # 是否只打印外循环的信息
    opts['method'] = opts0.get('method', None) # 内循环使用的求解器

    # 针对内循环的参数
    opts['gamma'] = float(opts0.get('gamma', 0.85))
    opts['rhols'] = float(opts0.get('rhols', 1e-6)) # 线搜索的参数
    opts['eta'] = float(opts0.get('eta', 0.2)) # 线搜索的参数
    opts['Q'] = float(opts0.get('Q', 1)) # 线搜索的参数

    return opts

# 增广拉格朗日函数法默认参数
def ALM_dual_optsInit(opts0: dict = {}):
    opts = {}
    opts['sigma'] = int(opts0.get('sigma', 10)) # 二次罚函数系数
    opts['maxit'] = int(opts0.get('maxit', 100))
    opts['maxit_inn'] = int(opts0.get('maxit_inn', 300))
    opts['thre'] = float(opts0.get('thre', 1e-6))
    opts['thre_inn'] = float(opts0.get('thre_inn', 1e-3))
    return opts

# 交换方向乘子法（对偶问题）默认参数
def ADMM_dual_optsInit(opts0: dict = {}):
    opts = {}
    opts['sigma'] = int(opts0.get('sigma', 10))
    opts['maxit'] = int(opts0.get('maxit', 1000))
    opts['thre'] = float(opts0.get('thre', 1e-6))
    return opts

# 交换方向乘子法（原问题）默认参数
def ADMM_primal_optsInit(opts0: dict = {}):
    opts = {}
    opts['sigma'] = int(opts0.get('sigma', 10))
    opts['maxit'] = int(opts0.get('maxit', 3000))
    opts['thre'] = float(opts0.get('thre', 1e-6))
    return opts

# 初始化内循环（具体求解器）参数
def optsInnerInit(opts: dict = {}):
    optsInner = {}
    optsInner['maxit_inn'] = int(opts.get('maxit_inn', 200)) # 内循环最大迭代次数 最大迭代次数，由 opts.maxit_inn 给出
    optsInner['ftol'] = float(opts.get('ftol', 1e-8)) # 针对函数值的停机判断条件 
    optsInner['gtol'] = float(opts.get('gtol', 1e-6)) # 针对梯度的停机判断条件
    optsInner['alpha0'] = float(opts.get('alpha0', 1)) #初始步长
    optsInner['gamma'] = float(opts.get('gamma', 0.9)) 
    optsInner['rhols'] = float(opts.get('rhols', 1e-6)) # 线搜索的参数
    optsInner['eta'] = float(opts.get('eta', 0.2)) # 线搜索的参数
    optsInner['Q'] = float(opts.get('Q', 1)) # 线搜索的参数
    return optsInner

def printAllDefaultOpts():
    print(f"testData: {testDataParams()}")
    print(f"gl_SGD_primal: {SGD_primal_optsInit()}") # 次梯度法默认参数
    print(f"gl_ProxGD_primal: {ProxGD_primal_optsInit()}") # 近似点梯度法默认参数
    print(f"gl_FProxGD_primal: {FProxGD_primal_optsInit()}") # 快速近似梯度法默认参数
    print(f"gl_ALM_dual: {ALM_dual_optsInit()}") # 增广拉格朗日函数法默认参数
    print(f"gl_ADMM_dual: {ADMM_dual_optsInit()}") # 交换方向乘子法（对偶问题）默认参数
    print(f"gl_ADMM_primal: {ADMM_primal_optsInit()}") # 交换方向乘子法（原问题）默认参数

# 初始化【结果输出】
def outInit():
    out = {}
    out['f_hist_outer'] = [] #  外循环每一次迭代的目标函数值
    out['f_hist_inner'] = [] # 每一步迭代的当前目标函数值（对应于当前的 μt）
    out['f_hist_best'] = [] # 每一步迭代的当前目标函数历史最优值（对应于当前的 μt）
    out['g_hist'] = [] # 可微部分梯度范数的历史值
    out['itr'] = 0 # 外层迭代次数
    out['itr_inn'] = 0 # 总内层迭代次数
    out['iters'] = None # zip格式记录每一次迭代的目标函数值
    out['fval'] = 0 # 最终目标函数数值
    # out['OptTime'] = 0
    out['flag'] = False # 标记是否收敛
    return out

# 使用连续化策略的外循环
def LASSO_group_con(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu0: float, opts: dict = {}):
    eigs = np.linalg.eig(np.matmul(A.T, A))[0]
    eigs = np.real(eigs[np.isreal(eigs)])
    # 初始化内循环（具体求解器）参数
    optsInner = optsInnerInit(opts)
    optsInner['alpha0'] = 1 / np.max(eigs)
    optsInner['mu0'] = mu0
    optsInner['ftol'] = opts['ftol'] * opts['ftol_init_ratio'] # 针对第一次循环放低停机准则
    optsInner['gtol'] = opts['gtol'] * opts['gtol_init_ratio'] # 针对第一次循环放低停机准则
    # 初始化【结果输出】
    outResult = outInit()

    x = x0
    mu_t = opts['mu1']
    f = objFun(x, A, b, mu_t)
    solver = opts['method']
    logger.debug(f"solver: {solver} solver_name: {solver.__name__}")
    logger.info(f"optsOuter: \n{opts}")
    logger.info(f"optsInner: \n{optsInner}")

    for k in range(opts['maxit']):
        logger.debug(f"--->iter {k} : current mu_t: {mu_t}<---")
        logger.debug(f"current fval: {f}")
        logger.debug(f"current alpha0: {optsInner['alpha0']}")
        # 不断严格化停机准则
        optsInner['gtol'] = max(optsInner['gtol'] * opts['etag'], opts['gtol']) # 保证下界
        optsInner['ftol'] = max(optsInner['ftol'] * opts['etaf'], opts['ftol']) # 保证下界
        logger.debug(f"optsInner['ftol']: {optsInner['ftol']}")
        logger.debug(f"optsInner['gtol']: {optsInner['gtol']}")
        
        # 启动内部求解器
        if not callable(solver):
            logger.error(f"optsOuter['method'] is not callable")
            raise ValueError(f"optsOuter['method'] is not callable")
        fp = f
        x, itr_inn, outInner = solver(x, A, b, mu_t, optsInner)
        f = outInner['f_hist_inner'][-1]
        outResult['f_hist_inner'].extend(outInner['f_hist_inner'])
        outResult['f_hist_outer'].append(f)

        r = np.matmul(A, x) - b
        # 由于L1-范数不可导 这里 nrmG 表示 LASSO 问题的最优性条件的违反度
        nrmG = np.linalg.norm(x - prox(x - np.matmul(A.T, r), mu0), ord="fro")
        logger.debug(f"current nrmG: {nrmG}")
        logger.debug("current abs(f-fp): {}".format(abs(f-fp)))
        logger.debug(f"current itr_inn: {itr_inn}")
        logger.debug(f"is_inner_converged: {outInner['flag']}")

        # flag 默认为false 默认每次外循环均缩小mu_t
        # 规定次数内，内循环迭代收敛达到停机条件 则不缩小mu_t
        if outInner['flag']:
            mu_t = max(mu_t * opts['factor'], mu0)
    
        outResult['itr_inn'] = outResult['itr_inn'] + itr_inn # 累加内层迭代总迭代数量

        if mu_t == mu0 and (nrmG < opts['gtol'] or abs(f-fp) < opts['ftol']):
            logger.debug(f"--->fval has converged to {f}")
            logger.debug(f"--->nrmG has converged to {nrmG}")
            logger.debug(f"--->abs(f-fp) has converged to {abs(f-fp)}")
            # 虽然最后一次迭代相比上次迭代没有更多精进，可舍去 但是迭代量确实计算了
            # outResult['itr_inn'] = outResult['itr_inn'] - itr_inn
            # 如果舍去外循环最后一次迭代，zip会以更短的 outResult['itr_inn'] 为主
            break
    
    outResult['fval'] = f # 最终目标函数值
    outResult['itr'] = k + 1 # 外层循环迭代次数
    logger.debug(f"len(outResult['f_hist_inner']): {len(outResult['f_hist_inner'])}")
    logger.debug(f"outResult['itr_inn']: {outResult['itr']}")
    logger.debug(f"--->end of LASSO_group_con<---")
    
    # 是否只使用外循环迭代的信息
    if opts['is_only_print_outer']:
        outResult['iters'] = zip(range(outResult['itr']), outResult['f_hist_outer'])
        return x, outResult['itr'], outResult
    else:
        outResult['iters'] = zip(range(outResult['itr_inn']), outResult['f_hist_inner'])
        return x, outResult['itr_inn'], outResult