# <div align=center>🎚️<br>Group-LASSO-problem 算法作业</div>

本项目为 2023 年秋季学期【最优化方法】课程的编程作业。
相关的课程信息链接如下：[课程页面](http://faculty.bicmr.pku.edu.cn/~wenzw/opt-2023-fall.html)、[作业描述](http://faculty.bicmr.pku.edu.cn/~wenzw/opt2015/homework5g.pdf)、[提交要求](http://faculty.bicmr.pku.edu.cn/~wenzw/opt2015/homework5-req.pdf)，后两个文件在项目 `/doc` 路径下有pdf版本。

## 代码运行

使用以下命令测试所有算法脚本，打印求解情况并绘制目标函数下降曲线：

```bash
python Test_group_lasso.py -S all
```

可以使用 `python Test_group_lasso.py -h` 查看帮助信息：
- `--solver` `-S` 可以指定某个求解器，默认为 `['gl_cvx_gurobi', 'gl_cvx_mosek']` 这两个求解器，也可以指定 `all` 运行所有求解器。
- `-mu` 可以指定 `mu` 的值，默认为 `1e-2`。
- `-seed` `-RS` 可以指定随机种子，默认为 `97108120`。
- `--plot` `-P` 可以指定是否绘制图像，默认为 `True`。

## 求解器函数说明

在本项目中，每个求解器的脚本名与函数名相同，脚本名称均为 `gl_*.py` 格式，函数名称均为 `gl_*()` 格式。一个样例函数接口形式如下：

```python
[x, iter, out] = gl_cvx_mosek(x0, A, b, mu, opts)
```
输入分别为给定的初始解 `x0` ，而 `A` 、 `b` 、 `mu` 是给定的数据。
输出 `x` 为算法求解出的解，`iter` 为输出为 `x` 时所对应的算法迭代次数。 `out` 为算法输出的其他信息，是一个字典结构，包含以下内容：
- `out['status']` 为算法求解状态，可以为 `optimal` 、 `infeasible` 、 `unbounded` 、 `other` 。
- `out['obj']` 为算法求解出的目标函数值。
- `out['iters']` 为算法每一步迭代的目标函数值与迭代号的组合列表。

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

项目可以使用 `conda` 或者 `pip` 安装依赖，但是 `mosek` 和 `gurobipy` 需要配置许可证书，具体方法参考官方文档[【Mosek installation】](https://docs.mosek.com/latest/install/installation.html)和[【gurobipy installation】](https://support.gurobi.com/hc/en-us/articles/360044290292)。

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
