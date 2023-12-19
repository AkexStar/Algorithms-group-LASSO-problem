import sys
import os
import re
import time
import logging
import numpy as np

class utils:
    def __init__(self):
        pass

def setLoggerLevel(logger, level: str):
    """设置日志级别
    Args:
        - _level (str): 日志级别，可选值为DEBUG, INFO, WARNING, ERROR, CRITICAL，默认为INFO
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('输入不正确的日志级别: %s' % level)
    logger.setLevel(numeric_level)
    logger.info(f"日志级别设置为{level.upper()}")

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
    logger.info(f"日志文件保存在: {current_work_dir}\logs\{name}-{now}.log")
    return logger, loggerName, cvxLogsName

logger, loggerName, cvxLogsName = loggerInit('utils')

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

reg_solver = {
    'GUROBI': re_iterc_default,
    # ([\s\S]{26})\:( +)(\d{1,2}) ([\s\S]{38})( +)([\-\+0-9\.eE]+)
    # 正则表达式赛高 排除前26个字符，匹配冒号，匹配空格，匹配1-2个数字，排除前38个字符，匹配空格，匹配数字，匹配小数点，匹配e，匹配E，匹配+，匹配-，匹配数字
    'MOSEK': re.compile(r'^ *([\s\S]{26})\:( +)(?P<iterc>\d{1,2}) ([\s\S]{38})( +)(?P<objv>[\-\+0-9\.eE]+)', re.MULTILINE), 
    'MOSEK_OLD': re.compile(r'^ *(?P<iterc>\d{1,3})\:?( +(?:[0-9\.eE\+\-]+)){4} +(?P<objv>[0-9\.eE\+\-]+)', re.MULTILINE),   # skip four columns
    'CVXOPT': re_iterc_default,
}

def cleanUpLog():
    with open(cvxLogsName,'w+',encoding='utf-8') as test:
        test.truncate(0)
        for line in test.readlines():
            line.replace(r'\0', '')

def parse_iters(s, solver=None):

    re_iterc = reg_solver[solver] if solver in reg_solver else re_iterc_default
    ret = []
    for match in re_iterc.finditer(s):
        ret.append((int(match.groupdict()['iterc']),
                    float(match.groupdict()['objv'])))
    return ret

def sparsity(x):
    logger.info(f"x.size: {x.size}")
    return np.sum(np.abs(x) > 1e-6 * np.max(np.abs(x))) / x.size
    # return np.sum(x <= 1e-5) / np.sum(np.ones_like(x))

def errX(x, x0):
    # return np.linalg.norm(x - u, ord='fro')
    # return np.linalg.norm(u - x, 'fro') / np.linalg.norm(u)
    return np.linalg.norm(x - x0, 'fro') / (1 + np.linalg.norm(x0, 'fro'))

def objFun(x, A, b, mu):
    r = np.dot(A, x) - b
    # return 0.5 * np.linalg.norm(A @ x - b, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
    return 0.5 * np.linalg.norm(r, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))

def errObj(obj, obj0):
    return np.abs(obj - obj0)

# 邻近算子
def prox(x, mu):
    nrmx = np.linalg.norm(x, ord=2, axis=1)
    flag = nrmx > mu
    prox_x = x - mu * x  / (nrmx.reshape(-1, 1) + 1e-10)
    prox_x = prox_x * flag.reshape(-1, 1)
    return prox_x

solversCollection = [
    # 'gl_cvx_gurobi',
    # 'gl_cvx_mosek',
    # 'gl_gurobi',
    # 'gl_mosek',
    'gl_SGD_primal', 
    # 'gl_ADMM_dual',
    # 'gl_ADMM_primal_direct', 
    # 'gl_ADMM_primal',
    # 'gl_FGD_primal', 
    # 'gl_FGD_primal_line_search',
    # 'gl_ProxGD_primal', 
    # 'gl_ProxGD_primal_line_search',
    # 'gl_FProxGD_primal', 
    # 'gl_FProxGD_primal_line_search',
    # 'gl_SGD_primal_normal_sgd',
    # 'gl_GD_primal', 
    # 'gl_GD_primal_normal_gd',
    # 'gl_gurobi_term',
]

def optsOuterInit(opts: dict):
    optsOuter = {}
    if opts is None:
        opts = {}
    optsOuter['maxit'] = opts.get('maxit', 100) # 最大迭代次数
    optsOuter['maxit_inn'] = opts.get('maxit_inn', 30) # 内循环最大迭代次数
    optsOuter['ftol'] = opts.get('ftol', 1e-10) # 针对函数值的停机判断条件 当相对函数值变化或相对历史最佳函数值变化小于该值时认为满足
    optsOuter['ftol_init_ratio'] = opts.get('ftol_init_ratio', 1e5) # 初始时停机准则 opts['ftol'] 的放大倍数
    optsOuter['gtol'] = opts.get('gtol', 1e-6) # 针对梯度的停机判断条件
    optsOuter['gtol_init_ratio'] = 1 / optsOuter['gtol'] # 初始时停机准则 opts['gtol'] 的放大倍数
    optsOuter['factor'] = opts.get('factor', 0.1) # 正则化系数的衰减率
    optsOuter['verbose'] = opts.get('verbose', False) # 是否打印每次迭代的信息
    optsOuter['mu1'] = opts.get('mu1', 100) # 初始的正则化系数（采用连续化策略，从更大的正则化系数开始）
    optsOuter['etaf'] = opts.get('etaf', 0.1) # 每步外层循环的停机判断标准 opts['ftol'] 的缩减
    optsOuter['etag'] = opts.get('etag', 0.1) # 每步外层循环的停机判断标准 opts['gtol'] 的缩减
    optsOuter['method'] = opts.get('method', None) # 内循环使用的求解器
    # for key in opts.keys():
    #     if key not in optsOuter.keys():
    #         optsOuter[key] = opts[key]
    return optsOuter

def optsInnerInit(opts: dict):
    optsInner = {}
    optsOuter = optsOuterInit(opts)
    optsInner['maxit'] = optsOuter['maxit_inn'] # 内循环最大迭代次数 最大迭代次数，由 opts.maxit_inn 给出
    optsInner['ftol'] = optsOuter['ftol']# * optsOuter['ftol_init_ratio'] # 针对函数值的停机判断条件 1e-5
    optsInner['gtol'] = optsOuter['gtol']# * optsOuter['gtol_init_ratio'] # 针对梯度的停机判断条件 1
    optsInner['alpha0'] = opts.get('alpha0', 1) #初始步长
    optsInner['mu0'] = opts.get('mu0', 1e-2) # 目标最小的mu0 便于连续化策略和内循环的求解器一起使用
    optsInner['gamma'] = opts.get('gamma', 0.85) # BB算法的参数
    optsInner['rhols'] = opts.get('rhols', 1e-6) # 线搜索的参数
    optsInner['eta'] = opts.get('eta', 0.2) # 线搜索的参数
    optsInner['Q'] = opts.get('Q', 1) # 线搜索的参数
    # for key in opts.keys():
    #     if key not in optsInner.keys():
    #         optsInner[key] = opts[key]
    return optsInner

def outInit():
    out = {}
    out['f_hist'] = [] # 每一步迭代的当前目标函数值（对应于当前的 μt）
    out['f_hist_best'] = [] # 每一步迭代的当前目标函数历史最优值（对应于当前的 μt）
    out['g_hist'] = [] # 可微部分梯度范数的历史值
    out['itr'] = 0 # 外层迭代次数
    out['itr_inn'] = 0 # 总内层迭代次数
    out['iters'] = None # zip记录每一次迭代的目标函数值
    out['fval'] = 0 # 最终目标函数数值
    # out['OptTime'] = 0
    out['flag'] = False # 标记是否收敛
    return out

# 使用连续化策略的外循环
def LASSO_group_con(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu0: float, opts: dict = {}):
    eigs = np.linalg.eig(np.matmul(A.T, A))[0]
    eigs = np.real(eigs[np.isreal(eigs)])
    # 初始化外循环（连续化策略）参数
    optsOuter = optsOuterInit(opts)
    # 初始化内循环（具体求解器）参数
    optsInner = optsInnerInit(optsOuter)
    optsInner['alpha0'] = 1 / np.max(eigs)
    optsInner['maxit'] = optsOuter['maxit_inn']
    optsInner['mu0'] = mu0
    # 初始化【结果输出】
    outResult = outInit()

    x = x0
    mu_t = optsOuter['mu1']
    f = objFun(x, A, b, mu_t)
    solver = optsOuter['method']

    logger.info(f"optsOuter: \n{optsOuter}")
    logger.info(f"optsInner: \n{optsInner}")

    for k in range(optsOuter['maxit']):
        logger.info(f"--->iter {k} : current mu_t: {mu_t}---<")
        logger.info(f"current fval: {f}")
        logger.info(f"current alpha0: {optsInner['alpha0']}")
        optsInner['gtol'] = max(optsInner['gtol'] * optsOuter['etag'], optsOuter['gtol'])
        optsInner['ftol'] = max(optsInner['ftol'] * optsOuter['etaf'], optsOuter['ftol'])
        logger.info(f"optsInner['ftol']: {optsInner['ftol']}")
        logger.info(f"optsInner['gtol']: {optsInner['gtol']}")
        
        fp = f
        x, itr_inn, outInner = solver(x, A, b, mu_t, optsInner)
        f = outInner['f_hist'][-1]
        outResult['f_hist'].extend(outInner['f_hist'])
        # logger.info(f"outResult['f_hist']=\n{outResult['f_hist']}")

        r = np.matmul(A, x) - b
        # 由于L1-范数不可导 这里 nrmG 表示 LASSO 问题的最优性条件的违反度
        nrmG = np.linalg.norm(x - prox(x - np.matmul(A.T, r), mu0), ord="fro")

        if not outInner['flag']:
            mu_t = max(mu_t * optsOuter['factor'], mu0)

        if mu_t == mu0 and (nrmG < optsOuter['gtol'] or abs(f-fp) < optsOuter['ftol']):
            break
        
        # logger.info(f"{outResult['itr']}")
        outResult['itr_inn'] = outResult['itr_inn'] + itr_inn # 累加内层迭代总迭代数量

    outResult['fval'] = f # 最终目标函数值
    outResult['itr'] = k + 1 # 外层循环迭代次数
    outResult['iters'] = zip(range(outResult['itr_inn']), outResult['f_hist'])
    # x, y = zip(*outResult['iters'])
    # logger.info(f"outResult=\n{x}\n{y}")

    return x, outResult['itr_inn'], outResult