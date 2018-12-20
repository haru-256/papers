# SVM の最適化手法（SMO）について

主にカーネルSVMの最適化手法であるSMO についてまとめる．基本的には [竹内一郎／烏山昌幸・著 「サポートベクトルマシン」](https://www.kspub.co.jp/book/detail/1529069.html) に沿って勉強した結果をまとめる．


ここではSVM において解 $\boldsymbol{\alpha}$  が最適解である必要十分条件についてまとめます．

## 前提知識

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



# SVM の最適化条件について（p 87~）



# SMOアルゴリズム（p 102~）

SVM は凸最適化問題として定式化されるため，小規模なデータセットに対しては第６章で学んだ汎用的な凸最適化手法を使うことができます．しかし，大規模なデータセットに対しては，SVMに特化した最適化手法を使うことで大幅に高速化させることができます．以下ではSVMの双対問題を解くための最適化手法を説明します[^1]．
## 分割法

## SMO

### ２変数の最適化



### ２変数の選択

2変数 $\beta_s, \beta_t$ の選択方法は様々である．前提として，各ステップでこれらの２変数をランダムに選んだとしてもSMOアルゴリズムは最適解に収束することが知られている．ただし，うまく$\beta_s, \beta_t$ を選部ことにより収束の速さを改善できることが知られている． 2変数の選択はヒューリスティックであり，実験的に速さが改善されることが知られており，証明はできない．しかし，前提があるため必ず収束する．

２変数の選択の方法はランダム以外に複数あるが，本書で紹介された手法を説明する．




[^1]: カーネルSVMでは主問題を解くことができないため．
>>>>>>> 6f601e2de6b32dca0422e4cab604bf4897c19cb2


# 参考文献

- 竹内一郎／烏山昌幸・著 「サポートベクトルマシン」
  https://www.kspub.co.jp/book/detail/1529069.html
- Y, Suhara「SMO徹底入門」
  https://www.slideshare.net/sleepy_yoshi/smo-svm