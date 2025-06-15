# 求导数的数值方法

对于一个函数$f(x)$，其定义域内非边界的一点$x\in A$，$f(x)$的导数，即瞬时变化率表示为
$$f'(x)=\lim_{h\rightarrow 0}\frac{f(x+h)-f(x)}{h}$$
直接使用该公式，就是向前差分法（程序`devint.py`中的`FORWARD`）。根据此定义，还可以定义出向后差分法
$$f'(x)=\lim_{h\rightarrow 0}\frac{f(x)-f(x-h)}{h}$$
它们的精度都是$o(h)$，但计算简便。
更精确地，可以同时取向前或向后的一点，即
$$f'(x)=\lim_{h\rightarrow 0}\frac{f(x+h)-f(x-h)}{2h}$$
这是中心差分法，精度为$o(h^2)$，是函数`derivative_func`的默认方法。
如果对精度的要求更高，还可以使用下面的方法：
$$f'(x) \approx \frac{-f(x+2h) + 8 f(x+h) - 8 f(x - h) + f(x - 2h)}{12 h}$$
或
$$f'(x) \approx \frac{f(x-3h)-9f(x-2h)+45f(x-h)-45f(x+h)+9f(x+2h)-f(x+3h)}{60h}$$
它们分别是五点差分法和七点差分法（实际用了4个和6个点），精度分别为$o(h^4)$和$o(h^6)$，其中$o(h^6)$的误差基本已经小到浮点数类型的极限，可以满足绝大多数情况下的要求了。下面以五点差分法为例进行推导：
$f(x)$的泰勒展开取到四阶：
$$f(x+kh)=f(x)+khf'(x)+\frac{(kh)^2}{2!}f''(x)+\frac{(kh)^3}{3!}f^{(3)}(x)+\frac{(kh)^4}{4!}f^{(4)}(x)+o(h^4)$$
我们线性组合$k=\pm 1,\pm 2, 0$的表达式，来表示出$f'(x)$，为了使精度达到$o(h^4)$，就要把零阶、二阶、三阶、四阶导数项全部化为0，一阶导数项为1，那么就等价于解线性方程组：
$$\begin{bmatrix}1&1&1&1&1\\2&1&-1&-2&0\\4&1&1&4&0\\8&1&-1&-8&0\\16&1&1&16&0\end{bmatrix}\begin{bmatrix} a \\ b\\ c \\ d \\ e\end{bmatrix}=\begin{bmatrix}0\\ \frac{1}{h}\\0\\0\\0 \end{bmatrix}$$
即可解出$f'(x)\sim af(x+2h)+bf(x+h)+cf(x-h)+df(x-2h)+ef(x)$的各项系数，得到导数公式。理论上用这样的方法可以无穷逼近函数导数的真值，只要取足够高阶的泰勒展开并解线性方程组。但由于Python浮点数的位数有限，使用更高阶的算法已经没有意义。

在函数的默认设置中，步长$h$是根据函数自变量自适应调节的：
$$h=\epsilon^\frac{1}{p+1}\max(|x|,1)$$
其中$\epsilon$的取值是`np.finfo(float).eps`，即浮点数误差；$p$是求导方法的精度$o(h^p)$。
# 求积分的数值方法
定积分的定义为$$\int_a^bf(x)\text{d}x=\lim_{\max(x_i-x_{i-1})\rightarrow0}\sum_{i=1}^{n}f(\xi_i)(x_i-x_{i-1})$$
取$\xi_i=\frac{x_i+x_{i-1}}{2}$，就是`integration_func`中的矩形法，精度$o(\Delta x)$。如果取用$x_i-x_{i-1}$作为高，$f(x_{i-1})$和$f(x_i)$作为梯形的底，用梯形面积的和作为积分值，它的精度就是$o((\Delta x)^2)$。
更精确的版本：取$f(x)$在区间$[x_{i-1},x_i]$上均匀的五个点：$\tilde{x}_j=x_{i-1},x_{i-1}+\frac{\Delta x_i}{4},x_{i-1}+\frac{\Delta x_i}{2},x_{i-1}+\frac{3\Delta x_i}{4},x_{i}$，构造Lagarange插值多项式：$$L(x)=\sum_{j=0}^kf(\tilde{x}_j)L_j(x), L_j(x)=\prod_{i=0, i\neq j}^k\frac{x-x_i}{x_j-x_i}$$
用该插值多项式计算积分：
$$\int_{x_{i-1}}^{x_i} f(x)\text{d}x=\sum_{j=0}^kf(\tilde{x}_j)\int_{x_{i-1}}^{x_i}L_j(x)\text{d}x$$
Lagarange多项式可以具体算出，积分得：
$$\int_{x_{i-1}}^{x_i} f(x)\text{d}x\approx\frac{\Delta x}{90}(7f(\tilde{x}_0)+32f(\tilde{x}_1)+12f(\tilde{x}_2)+32f(\tilde{x}_3)+7f(\tilde{x}_4))$$
它的误差是$o((\Delta x)^7)$。
# 调用与测试结果
`derivative_func`和`integration_func`的调用方法类似，在Python内自定义一个一元函数（也可以是具有`__call__`方法的类，如`interpolation.py`中的插值函数类），给定自变量取值（或积分上下限），用`INTEGRATION_TYPE`或`DERIVATIZATION_TYPE`枚举类指定积分或求导的计算方法，函数就会返回积分或求导的结果。积分示例如下：
```
def test_func(x:float):
    return np.exp(-x)+np.sin(x)-x**2

print(f"Rectangle:{intergration_func(-2,3,test_func,INTEGRATION_TYPE.RECTANGLE,n_domains=1000,absolute=False)}")
print(f"Trapzoid:{intergration_func(-2,3,test_func,INTEGRATION_TYPE.TRAPZOID,n_domains=1000,absolute=False)}")
print(f"Boole:{intergration_func(-2,3,test_func,INTEGRATION_TYPE.BOOLE,n_domains=20,absolute=False)}")
```
直接用原函数求得的积分真值是-3.75355197605，三种方法的返回值分别是：
```
Rectangle:-3.797682101220736
Trapzoid:-3.7976922419259216
Boole:-3.753551975199182
```
求导示例如下：
```
print(f"Forward:{derivative_func(3.0,test_func,DERIVATIZATION_TYPE.FORWARD)}")
print(f"Backward:{derivative_func(3.0,test_func,DERIVATIZATION_TYPE.BACKWARD)}")
print(f"Central:{derivative_func(3.0,test_func,DERIVATIZATION_TYPE.CENTRAL)}")
print(f"4th-order:{derivative_func(3.0,test_func,DERIVATIZATION_TYPE.FOUR_POINTS)}")
print(f"6th-order:{derivative_func(3.0,test_func,DERIVATIZATION_TYPE.SIX_POINTS)}")
```
代入导函数求得的真值是-7.03977956497，五个函数的返回值分别是：
```
Forward:-7.039779583613078
Backward:-7.039779543876648
Central:-7.039779564935732
4th-order:-7.0397795649679935
6th-order:7.039779564967933
```