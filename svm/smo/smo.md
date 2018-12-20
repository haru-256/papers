# SVM の最適化手法（SMO）について

主にカーネルSVMの最適化手法であるSMO についてまとめる．基本的には [竹内一郎／烏山昌幸・著 「サポートベクトルマシン」](https://www.kspub.co.jp/book/detail/1529069.html) に沿って勉強した結果をまとめる．

ここではSVM において解 $\boldsymbol{\alpha}$  が最適解である必要十分条件についてまとめます．

# はじめに
SVM の最適化問題は目的関数が凸な２次関数，制約式が１次関数からなる凸２次最適化問題（２次計画問題）です．凸２次最適化問題は比較的扱いやすいです．目的関数が凸な２次関数のため，どんな初期値から初めたとしても大域的最適解を得ることができます（というか，極小値が最小値なため局所的最適解が存在しない）．


# 前提知識

[竹内一郎／烏山昌幸・著 「サポートベクトルマシン」](https://www.kspub.co.jp/book/detail/1529069.html) の1.5章まで（~p22）を前提知識とします．



非線形SVMの主問題と双対問題

- 主問題
$$
\begin{align}
\begin{split}
\min_{\boldsymbol{w}, b, \boldsymbol{\epsilon}} &\ \ \frac{1}{2} \|\boldsymbol{w}\|^2 + C\sum_{i \in [n]} \xi \\
s.t.&\  -\Big(y_i (\boldsymbol{w}^{\top} \boldsymbol{\phi}(\boldsymbol{x}_i )+ b -1 + \xi_i\Big) \leq 0, \quad i\in[n] \\
& - \xi_i \leq 0, \quad i \in [n]
\end{split}
\end{align}
$$
- 双対問題
$$
\begin{align}
\begin{split}
\max_{\boldsymbol{\alpha}} &\ \ - \frac{1}{2}\sum_{i,j \in [n]} \alpha_i \alpha_j y_i y_j  K(\boldsymbol{x}_i, \boldsymbol{x}_j) + \sum_{i \in [n]} \alpha_i \\
s.t.&\  \sum_{i \in [n]}\alpha_i y_i = 0\\
& 0 \leq \alpha_i \leq C,\quad i \in [n]
\end{split}
\end{align}
$$

- KKT条件
$$
\begin{align}
\frac{\partial L}{\partial \boldsymbol{w}} = \boldsymbol{w} - \sum_{i \in [n]} \alpha_i y_i \boldsymbol{\phi}(\boldsymbol{x}_i)&= \boldsymbol{0} \\

\frac{\partial L}{\partial b} = - \sum_{i \in [n]} \alpha_i y_i &= 0 \\
 
\frac{\partial L}{\partial b} = C-\alpha_i -\mu_i &= 0 \\

 -\Big(y_i (\boldsymbol{w}^{\top} \boldsymbol{\phi}(\boldsymbol{x}_i )+ b -1 + \xi_i\Big) &\leq 0, \quad i\in[n]\\
 
 -\xi_i &\leq 0, \quad i \in[n]\\
 \alpha_i &\geq 0,\quad i\in[n]\\
 \mu_i &\geq 0, \quad i\in[n]\\
 \alpha_i \Big(y_i (\boldsymbol{w}^{\top} \boldsymbol{\phi}(\boldsymbol{x}_i )+ b -1 + \xi_i\Big) &= 0, \quad i\in[n]\\
 \mu_i \xi_i &= 0, \quad i\in[n]\\
 
\end{align}
$$

# SVM の最適化条件について（p 87~）

３つの添え字集合

$$
\begin{align*}
\mathcal{O} &= \{ i \in [n]\ |\  \alpha_i=0 \} \\
\mathcal{M} &= \{ i \in [n]\ |\  0 \leq\alpha_i \leq C\} \\
\mathcal{L} &= \{ i \in [n]\ |\  \alpha_i=C \} \\
\end{align*}
$$

を定義し，$f(\boldsymbol{x}) = \sum_{i\in[n]} \alpha_i y_i K(\boldsymbol{x}_i, \boldsymbol{x}) + b$ とすると，以下の条件がSVMのKKT条件の必要十分条件となります．
$$
\begin{align*}
y_i f(\boldsymbol{x}_i) &\geq 1, \quad i\in \mathcal{O} \\
y_i f(\boldsymbol{x}_i) &= 1, \quad i\in \mathcal{M} \\
y_i f(\boldsymbol{x}_i) &\leq 1, \quad i\in \mathcal{L} \\
\boldsymbol{y}^{\top} \boldsymbol{\alpha} &= 0 \\
\boldsymbol{0} \leq \boldsymbol{\alpha}&\leq C\boldsymbol{1}
\end{align*}
$$

KKT条件と比べると $\mu_i$ と $\xi_i$ が消され，より扱いやすくなっています．これが解の最適性として使われます．



# SMOアルゴリズム（p 102~）

SVM は凸最適化問題として定式化されるため，小規模なデータセットに対しては，汎用的な凸最適化手法（内点法など）を使うことができます．しかし，大規模なデータセットに対しては，SVMに特化した最適化手法を使うことで大幅に高速化させることができます．以下ではSVMの双対問題を解くための最適化手法を説明します[^1]．
## 分割法



## SMO

### ２変数の最適化



### ２変数の選択

2変数 $\beta_s, \beta_t$ の選択方法は様々である．前提として，各ステップでこれらの２変数をランダムに選んだとしてもSMOアルゴリズムは最適解に収束することが知られている．ただし，うまく$\beta_s, \beta_t$ を選部ことにより収束の速さを改善できることが知られている． 2変数の選択はヒューリスティックであり，実験的に速さが改善されることが知られており，証明はできない．しかし，前提があるため必ず収束する．

２変数の選択の方法はランダム以外に複数あるが，本書で紹介された手法を説明する．




[^1]: カーネルSVMでは主問題を解くことができないため双対問題を解くしかない．しかし，線形SVMの場合，主問題と双対問題のどちらを解いても良く，主問題と双対問題の関係をうまく利用することでカーネルSVMよりも効率的な学習のできるアルゴリズム（DCDM）が開発されている．


# 参考文献

- 竹内一郎／烏山昌幸・著 「サポートベクトルマシン」
  https://www.kspub.co.jp/book/detail/1529069.html
- Y, Suhara「SMO徹底入門」
  https://www.slideshare.net/sleepy_yoshi/smo-svm