# Softmax 回归

> 注：一下 $log$ 均可用 $ln$ 代替

## 回归 vs 分类

- 回归估计连续值
- 分类预测离散类别：分多少类就是几类问题

回归

- 单连续值输出
- 自然区间 R
- 跟真实值区别做损失

分类

- 多个输出，输出个数是类别个数
- 输出 i 是预测为第 i 类的置信度

## 均方损失

- 对类别进行 one-hot 编码，设定一个向量 $y$，若类别是第 i 类（概率最大），则 $y_i = 1$ 其他的都为 0，如下：

  $y = [y_1, y_2, ..., y_n]^T$

  $y = [0,...1,...,0]$

  $y_i =\begin{cases}
    1 & \text{ if } i= y \\
    0 & \text{ if } otherwise
  \end{cases}$

  

- 使用均方损失训练

- 最大值z

- 最大值作为预测，作为结果的置信度要远远大于其他类，即$o_y-o_i >= \delta(y, i)$，这里 $\delta$ 是一个阈值


  $\hat y = argmax o_i$

## 校验比例

要将概率作为输出，而不是向量，这里经过一个 `softmax(o)` 函数 $\hat y = softmax(o)$ ，将原本的向量改为概率向量输出，其中的每一项做如下转换
$$
\hat y_i = \frac {exp(o_i)} {\sum_k exp(o_k)} = \frac {e^{o_i}} {e^{o_1} + ... + e^{o_i} + ... +e^{o_n}}
$$

对 $y$ 也要如此处理，最后以两者的区别作为损失

## Softmax 和 交叉熵损失

- 交叉熵越小，两个概率模型越接近：$H(p,q) = \sum_i{-p_ilog(q_i)}=\sum_i{-p_iln(q_i)}$

- 损失函数：$\ell(y,\hat y) = \sum_i{-y_ilog(y_i)} = -log{\hat y_y} = -ln{\hat y_y}$，这里的 $\hat {y_y}$ 是指预测值中对应的真实值的位置，比如 $y = [0,0,1]$，那么 $\hat {y_y}$ 对应的就是$\hat y$ 第三个位置，只关心正确类的预测值的置信度

-  梯度是真实概率和预测概率的区别：$\partial_oi \ell(y,\hat y) = softmax(o)_i - y_i$

  > $$
  > \partial_{o_i} \ell(y,\hat y) = 
  > \partial_{o_i}(-ln{\hat y_i}) =
  > \partial_{o_i}-ln(\frac {exp(o_i)} {\sum_k exp(o_k)}) =
  > softmax(o)_i - y_i
  > $$

## 补充：损失函数

1. 均方损失 $\frac 1 2 (y - \hat y)^2$
2. L1 损失 $| y- \hat y|$
3. 鲁棒损失