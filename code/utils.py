import sys
import os
import re
import time
import logging
import numpy as np

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
    logging.basicConfig(filename = loggerName,
                        level = logging.INFO,
                        encoding='utf-8',
                        format = '[%(asctime)s] %(filename)s: %(funcName)s: %(levelname)s: %(message)s')
    logger.info(f"日志文件保存在: {current_work_dir}\logs\{name}-{now}.log")
    return logger, loggerName

logger, loggerName = loggerInit('utils')

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
    with open(r'./logs/gl_cvx.log','w+',encoding='utf-8') as test:
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
    # return np.sum(np.abs(x) > 1e-5) / x.size
    return np.sum(x <= 1e-5) / np.sum(np.ones_like(x))

def errFun(x, x0):
    # return np.linalg.norm(x - u, ord='fro')
    # return np.linalg.norm(u - x, 'fro') / np.linalg.norm(u)
    return np.linalg.norm(x - x0, 'fro') / (1 + np.linalg.norm(x0, 'fro'))

def testData(**opts):
    if "seed" in opts:
        seed = opts["seed"]
    else:
        seed: int = 97108120 # seed = ord("a") ord("l") ord("x")
    if "mu" in opts:
        mu = opts["mu"]
    else:
        mu = 1e-2
    np.random.seed(seed)
    n = 512
    m = 256
    A = np.random.randn(m, n)
    k = round(n * 0.1)
    l = 2
    p = np.random.permutation(n)[:k]
    u = np.zeros((n, l))
    u[p, :] = np.random.randn(k, l)
    b = A @ u
    # x0 = np.random.rand(n, l)
    # x0 = u + np.random.rand(n, l) * 0.001
    # x0 = np.zeros((n, l))
    x0 = np.random.randn(n, l)
    f_u = 0.5 * np.linalg.norm(A @ u - b, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(u, ord=2, axis=1))
    sparsity_u = sparsity(u)
    return x0, A, b, mu, u, f_u, sparsity_u
