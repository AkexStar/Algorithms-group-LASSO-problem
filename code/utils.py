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

def errFun(x, x0):
    # return np.linalg.norm(x - u, ord='fro')
    # return np.linalg.norm(u - x, 'fro') / np.linalg.norm(u)
    return np.linalg.norm(x - x0, 'fro') / (1 + np.linalg.norm(x0, 'fro'))

def objFun(x, A, b, mu):
    return 0.5 * np.linalg.norm(A @ x - b, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))

solversCollection = [
    'gl_cvx_gurobi',
    'gl_cvx_mosek',
    'gl_gurobi',
    'gl_mosek',
    # 'gl_SGD_primal', 
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
