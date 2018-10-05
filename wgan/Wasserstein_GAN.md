# Wasserstein GAN (Arjovsky et al. 12/06/2017)

https://arxiv.org/abs/1701.07875

## 1. どんなもの？

この論文ではGANの枠組みにおいて，Generatorが最適化するコストをEarth Mover (EM) 距離
$$
W(\mathbb{P}_r, \mathbb{P}_g) = \inf_{\gamma \in \Pi (\mathbb{P}_r, \mathbb{P}_g) } \mathbb{E}_{(x, y) \sim \gamma } [\| x - y\|]
$$
とする手法 Wasserstein GANを提案した．

## 2. 先行研究と比べてどこがすごい？
[Goodfellow](https://arxiv.org/abs/1406.2661)が提案したGenerative Adversarial Networks (GAN) では Jensen-Shannon Divergence
$$
JS(\mathbb{P}_r || \mathbb{P}_g) = \frac{1}{2} KL (\mathbb{P}_r || \mathbb{P}_m)  + \frac{1}{2} KL (\mathbb{P}_g || \mathbb{P}_m) \\
\mathbb{P}_m = (\mathbb{P}_r + \mathbb{P}_g ) / 2
$$
 を最小化するようにGeneratorを最適化していた．しかし，この論文ではGeneratorが最適化するコストをEarth Mover (EM) 距離
$$
W(\mathbb{P}_r, \mathbb{P}_g) = \inf_{\gamma \in \Pi (\mathbb{P}_r, \mathbb{P}_g) } \mathbb{E}_{(x, y) \sim \gamma } [\| x - y\|] \tag{1}
$$
とした．ここで，$\mathbb{P}_r$ はデータの生成分布，$ \mathbb{P}_g​$はサンプルの生成分布である．Discriminatorはデータの生成分布とサンプルの生成分布とのEM距離をはかり，Generatorはその距離を最小化するように最適化する．これにより，意味がありかつ連続な2つの確率分布の距離をGeneratorに与える．よってGeneratorの勾配がより良く（連続に）なり安定した学習が可能になった．また，Generatorの収束と生成画像の品質と関係のあるロスをCriticが提供する．よってハイパーパラメータの調整にロスをmetricとして扱える．しかし，Criticの構造が異なる場合では単純に比較することができないため，モデル構造の比較を行う指標とはなり得ない．


## 3. 技術や手法のキモはどこ？

式(1)のように可能なすべての同時分布を考慮するのは大変なので [Kantorovich-Rubinstein duality](https://vincentherrmann.github.io/blog/wasserstein/) より以下の双対問題を考える．
$$
W(\mathbb{P}_r, \mathbb{P}_{\theta}) = \sup_{\| f \|_L \leq 1 } \mathbb{E}_{x \sim \mathbb{P}_r } [f(x)] -  \mathbb{E}_{x \sim \mathbb{P}_\theta } [f(x)]
$$

ここで，$\mathbb{P}_\theta$ はGeneratorの出力分布（つまり，サンプルの生成分布）を表す．ここで $f$ は1-Lipschitz 関数である．しかし，式(2)の$\| f \|_L \leq 1$ を $\| f \|_L \leq K$ に置き換えたとしても，結局$K \cdot  W(\mathbb{P}_r, \mathbb{P}_{\theta})$ を考えることになる．よって，もしある $K$ についてすべてK-Lipschitzである，パラーメタ化された関数族 $\{f\}_{w \in \mathcal{W}}​$ が存在するとすると，以下の式を考えることができる．

$$
\begin{align}
\max_{\| w \in \mathcal{W} \|} \mathbb{E}_{x \sim \mathbb{P}_r } [f_{w}(x)] -  \mathbb{E}_{z \sim p(z) } [f_{w}(g_{\theta} (z))]
\end{align}
$$

ここで $f_w, g_\theta$ はそれぞれCritic, Generator を表す．また，その時のパラメータ $\theta$ の勾配は，

$$
\nabla_{\theta} W(\mathbb{P}_r, \mathbb{P}_{\theta}) = - \mathbb{E}_{z \sim p(z) } [\nabla_\theta f_{w}(g_{\theta} (z))]
$$

ただし，Critic $ f_{w}$ がK-Lipschitzとなるためには $\mathcal{W}$ がコンパクトでなければならない（コンパクトであれば $ f_{w}$ はある$K$に対してK-Lipschitzとなる）． $\mathcal{W}$ がコンパクトであるために，各勾配更新で重みを fixed box (e.g. $W = [-0.01, 0.01]^{l}$ ) 内へclampする．

WGANのアルゴリズムは以下の通りになる．

![](https://cdn-images-1.medium.com/max/2000/1*JOg9lC2JLl2Crmx5uk6S2g.png)

論文では書いていないが，著者は実装において以下（[Github](https://github.com/martinarjovsky/WassersteinGAN#a-few-notes)より）のことをした．
> The only addition to the code (that we forgot, and will add, on the paper) are the lines 163-166 of main.py. These lines act only on the first 25 generator iterations or very sporadically (once every 500 generator iterations). In such a case, they set the number of iterations on the critic to 100 instead of the default 5. This helps to start with the critic at optimum even in the first iterations. There shouldn’t be a major difference in performance, but it can help, especially when visualizing learning curves (since otherwise you’d see the loss going up until the critic is properly trained). This is also why the first 25 iterations take significantly longer than the rest of the training as well.

また，著者は入力画像を $[-1, 1]$ に正規化している．さらに $p(\boldsymbol{z})$を $\mathcal{N} (0, 1)$ としている． 

## 4. どうやって有効だと検証した？

実際に画像生成の実験を行い，WGANが通常のGANよりも学習が安定すること（Improved stability），またCriticのロスがGeneratorの収束とサンプルの質とに相関があること（Meaningful loss metrics）を実験的に示した．

**共通条件**

- 使用したデータは [LSUN-Bedrooms ](http://www.yf.io/p/lsun)
- 用いたベースラインは [DCGAN](https://arxiv.org/abs/1511.06434) （畳み込み構造を用い，Goodfellowが提案した通常のGANの最適化方法（$-\log{D} $ トリックを使用）を使用したモデル．）である．
- 生成画像は３チャネルの 64x64 画像である．
- すべての実験においてハイパーパラメータは Algorithm 1 で指定したものを使用.

### Meaningful loss metrics
**条件**

- 使用したモデルは以下の3つ
  1. DCGAN構造のWGAN
  2. DCGAN構造のWGANで，Generatorを各層が512unitsで活性化関数がReLUの総層数 4層のMLPに置き換えたモデル．
  3. DCGAN構造のWGANで，GeneratorとCritic を各層が512unitsで活性化関数がReLUの総層数 4層のMLPに置き換えたモデル．

- 比較モデル
  1. DCGAN構造のGAN
  2. DCGAN構造のGANで，Generatorを各層が512unitsで活性化関数がReLUの総層数 4層のMLPに置き換えたモデル．
  3. DCGAN構造のGANで，GeneratorとCritic を各層が512unitsで活性化関数がReLUの総層数 4層のMLPに置き換えたモデル．

**結果**
この実験では，Figure 3: のようにCriticのロスとGeneratorの収束とサンプルの質とに相関があることがわかった．同様のモデルで，Goodfellow 提案の最適化方法（JS がコスト）では Figure 4: のようにサンプルが良くなったとしても，JS 推定量は増加，または一定のままである．

**結論**
このことから，WGANでは Critic のロスとサンプルの質とに強い相関があり，ロスを基準にハイパーパラメータの調整などが行える．

*注意*
Critic のロスとサンプルの質とに強い相関があると言っても，他のモデルとの性能比較に使用できるわけではない．何故ならば， Wesserstein 距離の定数係数 $K$ の値はCritic のモデル自体に依存しているため，異なる構造のCriticとは比較することができない．

### Improved stability

WGANでは，Criticを最適値まで学習することができ，最適値ではCriticは単にGeneratorとデータの生成分布との距離を返すため，もはやGeneratorとDiscriminator（Critic）のバランスを取る必要がない．Criticがより良くなれば，Generatorによりhigh quality な勾配を与えることができる．


## 5. 議論はある？

WGANの最適化方法では，CriticのOptimizerにモーメンタム項を使用した最適化手法（e.g Adam: $\beta_1 > 0$）や高い学習率を設定すると学習がうまくいかないことが観測された．そのため著者は RMSPropを用いている．

これはCriticのロスが非定常であることが原因である．（これはcriticのパラメータ$w$がclampされるから？）



パラメータ$w$がCriticを最適化する各イテレーションのたびにclampされるのはかなり強引！！

## 6. 次に読むべき論文は？

[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
WGAN-gp を提案した論文．Criticのパラメータ$w$をclampせずにCriticがK-Lipschitzであるような制約を発見した．

## 参考文献

[From GAN to WGAN](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)

[Wasserstein GAN ご注文は機械学習ですか？闇の魔術に対する防衛術 ](http://musyoku.github.io/2017/02/06/Wasserstein-GAN/)

