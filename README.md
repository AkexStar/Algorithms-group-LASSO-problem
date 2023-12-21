# <div align=center>🎚️<br>Group-LASSO-problem 优化算法作业 </div>

本项目为 2023 年PKU秋季学期【最优化方法】课程的编程作业

相关的课程信息链接如下：[课程页面](http://faculty.bicmr.pku.edu.cn/~wenzw/opt-2023-fall.html)、[作业描述](http://faculty.bicmr.pku.edu.cn/~wenzw/opt2015/homework5g.pdf)、[提交要求](http://faculty.bicmr.pku.edu.cn/~wenzw/opt2015/homework5-req.pdf)，后两个文件在项目 [/doc](./doc) 路径下有pdf版本。

## 问题描述

本项目实现的算法面向以下问题：
```math
\min _{x \in \mathbb{R}^{n \times l}} \frac{1}{2}\|A x-b\|_F^2+\mu\|x\|_{1,2}
```
其中 $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^{m \times l}$, $\mu>0$ 并且
```math
\|x\|_{1,2}=\sum_{i=1}^n\|x(i, 1: l)\|_2
```
其中 $x(i, 1: l)$ 是矩阵 $x$ 的第 $i$ 行。

## 使用测试脚本

### 一览所有求解器

使用以下命令测试所有算法脚本，打印求解情况并绘制目标函数下降曲线：

```bash
python Test_group_lasso.py -S all -P
```

结果为
```txt
exact_fval=f_u: 0.652304376262884
sparsity_u: 0.099609375

Solver            Objective    Obj_ABS_Error    x_u_Error    Time(s)    Iter    Sparsity
--------------  -----------  ---------------  -----------  ---------  ------  ----------
cvx_mosek          0.652291      1.36211e-05  3.9124e-05   0.453096       13   0.102539
cvx_gurobi         0.652291      1.33551e-05  3.75525e-05  0.860177       12   0.102539
mosek              0.652301      3.33869e-06  9.84223e-06  0.33137        11   0.0996094
gurobi             0.652291      1.33727e-05  3.97062e-05  0.60077        13   0.103516
SGD_primal         0.652294      1.02855e-05  5.52482e-05  0.63127      1473   0.121094
ProxGD_primal      0.652291      1.36421e-05  3.89562e-05  0.187898      190   0.102539
FProxGD_primal     0.652291      1.36421e-05  3.8964e-05   0.249066      486   0.102539
ALM_dual           0.652308      3.97276e-06  6.83455e-05  0.303512       70   0.0996094
ADMM_dual          0.652309      4.80892e-06  6.84793e-05  0.0626595      86   0.0996094
ADMM_primal        0.652291      1.34764e-05  3.8987e-05   0.46301      2694   0.102539
```
![fval curve](https://github.com/AkexStar/Algorithms-group-LASSO-problem/assets/55226358/3f08db06-9d2f-472b-a430-fdabb4d87d15)

### 测试脚本的帮助

不知道如何使用测试脚本？可以使用 `python Test_group_lasso.py -h` 查看帮助信息：

```txt
usage: Test_group_lasso [-h] [--solvers SOLVERS [SOLVERS ...]] [--seed SEED] [--plot] [--log {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--opts OPTS [OPTS ...]] [--compare] [--version] [--printDefaultOpts]

测试不同的求解器进行group-lasso求解 https://github.com/AkexStar/Algorithms-group-LASSO-problem

optional arguments:
  -h, --help            show this help message and exit
  --version, -V         show program's version number and exit
  --solvers SOLVERS [SOLVERS ...], -S SOLVERS [SOLVERS ...]
                        指定求解器名称, 输入`all` 可以测试本项目中所有的求解器函数。默认填充为 `['gl_cvx_gurobi', 'gl_cvx_mosek']` 这两个求解器。
  --seed SEED, -RS SEED
                        指定测试数据的随机数种子。默认为97108120，为 `alx` 的ASCII码依次排列。
  --plot, -P            表明是否绘制迭代曲线，如果增加此参数，则绘制。
  --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}, -L {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        指定日志等级。默认为INFO。
  --opts OPTS [OPTS ...], -O OPTS [OPTS ...]
                        指定测试数据的参数，格式为`key=value`，可以有多个。例如 `-O gl_ALM_dual={'maxit':60, 'maxit_inn':100} testData={'m'=256, 'n':512}` 。没有指定的参数将使用默认值。
  --compare, -C         表明是否将计算得到的最优解与mosek和gurobi的结果比较，如果增加此参数，则比较。
  --printDefaultOpts, -PDO
                        展示所有默认opts参数。
```
### 内设参数和默认值

不知道测试脚本的 `--opts` 如何使用？可指定的参数有哪些？默认参数值是多少？可以使用 `python Test_group_lasso.py -PDO` 打印所有可设定参数和其默认值：

```txt
testData: {'seed': 97108120, 'mu': 0.01, 'n': 512, 'm': 256, 'l': 2, 'r': 0.1}
gl_SGD_primal: {'maxit': 50, 'maxit_inn': 250, 'ftol': 1e-09, 'ftol_init_ratio': 1000000.0, 'etaf': 0.1, 'gtol': 1e-06, 'gtol_init_ratio': 1000000.0, 'etag': 0.1, 'factor': 0.1, 'mu1': 10.0, 'is_only_print_outer': False, 'method': None, 'gamma': 0.9, 'rhols': 1e-06, 'eta': 0.2, 'Q': 1.0}
gl_ProxGD_primal: {'maxit': 50, 'maxit_inn': 250, 'ftol': 1e-09, 'ftol_init_ratio': 1000000.0, 'etaf': 0.1, 'gtol': 1e-06, 'gtol_init_ratio': 1000000.0, 'etag': 0.1, 'factor': 0.1, 'mu1': 10.0, 'is_only_print_outer': False, 'method': None, 'gamma': 0.85, 'rhols': 1e-06, 'eta': 0.2, 'Q': 1.0}
gl_FProxGD_primal: {'maxit': 50, 'maxit_inn': 250, 'ftol': 1e-09, 'ftol_init_ratio': 1000000.0, 'etaf': 0.1, 'gtol': 1e-06, 'gtol_init_ratio': 1000000.0, 'etag': 0.1, 'factor': 0.1, 'mu1': 10.0, 'is_only_print_outer': False, 'method': None, 'gamma': 0.85, 'rhols': 1e-06, 'eta': 0.2, 'Q': 1.0}
gl_ALM_dual: {'sigma': 10, 'maxit': 100, 'maxit_inn': 300, 'thre': 1e-06, 'thre_inn': 0.001}
gl_ADMM_dual: {'sigma': 10, 'maxit': 1000, 'thre': 1e-06}
gl_ADMM_primal: {'sigma': 10, 'maxit': 3000, 'thre': 1e-06}
```

### 测试脚本的调试

如果要进行调试，在使用vs code时建议使用以下配置 `launch.json`：

```json
{
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "stopOnEntry": false,
            "python": "${command:python.interpreterPath}",
            "program": "${file}",
            "cwd": "${workspaceRoot}",
            "env": {
                "PYTHONPATH": "${workspaceRoot}"
            },
        }
    ]
}
```

## 求解器函数说明

在本项目中，每个求解器的脚本名与函数名相同，脚本名称均为 `gl_*.py` 格式，函数名称均为 `gl_*()` 格式。一个样例函数接口形式如下：

```python
[x, iter, out] = gl_solver_name(x0, A, b, mu, opts)
```

输入分别为给定的初始解 `x0` ，而 `A` 、 `b` 、 `mu` 是给定的数据。
`opts` 是用来指定求解器内部参数的字典变量。其根据不同的求解器可填入不同的指定参数。参数种类和默认参数值可见文件[utils.py](./src/utils.py)中各个 `solver_name_optsInit()` 函数。

输出 `x` 为算法求解出的解，`iter` 为输出为 `x` 时所对应的算法迭代次数。 `out` 为算法输出的其他信息，是一个字典结构，包含以下内容：

- `out['fval']` 为算法求解出的目标函数值。
- `out['iters']` 为算法每一步迭代的目标函数值与迭代号的zip组合列表。
- 上述两项为调用mosek和gurobi的输出信息，主要从求解器的日志输出中用正则表达式爬取。而自行编写的求解器则除上述两项外具有更多记录信息。

如果想只测试某个算法函数，可参考以下代码：

```python
import src.utils as utils

dataOpts = {} # 指定生成测试数据的参数 正则化系数mu 随机数种子seed 矩阵大小m n l 非零元素占比r
# dataOpts = {'mu':0.01, 'seed':2333, 'm':256, 'n':512, 'l':2, 'r':0.1}
x0, A, b, mu, u, f_u = utils.testData(dataOpts) # 得到测试数据
solver_opts = {} # 求解算法参数
solver_opts['solver_name'] = 'gl_ALM_dual' # 指定使用哪个求解算法
x, iters_N, out = utils.testSolver(x0, A, b, mu, solver_opts)
```

或直接调用某个算法函数：

```python
import src.utils as utils
from src.gl_ALM_dual import *

x0, A, b, mu, u, f_u = utils.testData({})
x, iter, out = gl_ALM_dual(x0, A, b, mu, {})
```

## 项目文件结构
```txt
ALGORITHMS-GROUP-LASSO-PROBLEM
│  .gitignore
│  README.md
│  report.ipynb
│  requirements.txt
│  Test_group_lasso.py
│
├─doc
│      homework-description.pdf
│      homework-requirement.pdf
│
└─src
       gl_ADMM_dual.py
       gl_ADMM_primal.py
       gl_ALM_dual.py
       gl_cvx_gurobi.py
       gl_cvx_mosek.py
       gl_FGD_primal.py
       gl_FProxGD_primal.py
       gl_GD_primal.py
       gl_gurobi.py
       gl_mosek.py
       gl_ProxGD_primal.py
       gl_SGD_primal.py
       utils.py
       __init__.py
```

## 软件环境版本

| **Name** | **Version** |
| :------------: | :---------------: |
|    Windows    |  11+.0(22631.2)  |
|     python     |      3.9.18      |
|      pip      |      23.3.1      |
|     cvxpy     |       1.4.1       |
|    gurobipy    |      11.0.0      |
|     Mosek     |      10.1.21      |
|   matplotlib   |       3.8.2       |
|     numpy     |      1.26.2      |
|    tabulate    |       0.9.0       |

本项目使用 `conda` 管理环境，使用 `conda create -n glp python=3.9` 创建环境，使用 `conda activate glp` 激活环境，使用 `conda deactivate` 退出环境。

项目可以使用 `conda` 或者 `pip` 安装各项依赖包，但是其中 `mosek` 和 `gurobipy` 需要配置许可证书，具体方法参考官方文档【[Mosek installation](https://docs.mosek.com/latest/install/installation.html)】和【[gurobipy installation](https://support.gurobi.com/hc/en-us/articles/360044290292)】。

## 参考资料

- [1] [repo: group-lasso-optimization](https://github.com/gzz2000/group-lasso-optimization)
- [2] [课程提供的 Matlab 代码样例](http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/contents/contents.html)
- [3] [CVXPY 说明文档](https://www.cvxpy.org/index.html)
- [4] [python logging 说明文档](https://docs.python.org/3/howto/logging-cookbook.html)
- [5] [CVXPY stdout 输出重定向方法](https://stackoverflow.com/questions/68863458/modifying-existing-logger-configuration-of-a-python-package)
- [6] [python 正则表达式说明文档](https://docs.python.org/3/library/re.html)
- [7] [matplotlib 说明文档](https://matplotlib.org/stable/contents.html)

## 致谢

- 感谢 [@wenzw](http://faculty.bicmr.pku.edu.cn/~wenzw/) 老师提供的课程资料，本项目的部分函数实现参考了其内容。
- 感谢 [@gzz2000](https://github.com/gzz2000/) 公开的代码样例，本项目部分函数的实现参考了其内容。
- 感谢 [@zhangzhao2022](https://github.com/zhangzhao2022/) 提供的支持，本项目的部分函数实现受到其指导帮助。
- 感谢 [@cvxgrp](https://github.com/cvxgrp/) 提供的 CVXPY 优化库，本项目使用了其提供的接口。
