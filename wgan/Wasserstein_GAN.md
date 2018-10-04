# Wasserstein GAN (Arjovsky et al. 12/06/2017)

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
とした．ここで，$\mathbb{P}_r$ はデータの生成分布（目標），$ \mathbb{P}_g​$はサンプルの生成分布である．これにより，意味がありかつ連続な2つの確率分布の距離をGeneratorに与える．よってGeneratorの勾配がより良く（連続に）なり安定した学習が可能になった．また，Generatorの収束と生成画像の品質と関係のあるロスをCriticが提供する．よってハイパーパラメータの調整にロスをmetricとして扱える．しかし，Criticの構造が異なる場合では単純に比較することができないため，モデル構造の比較を行う指標とはなり得ない．


## 3. 技術や手法のキモはどこ？

式(1)のように可能なすべての同時分布を考慮するのは大変なので [Kantorovich-Rubinstein duality](https://vincentherrmann.github.io/blog/wasserstein/) より以下の双対問題を考える．
$$
W(\mathbb{P}_r, \mathbb{P}_{\theta}) = \sup_{\| f \|_L \leq 1 } \mathbb{E}_{x \sim \mathbb{P}_r } [f(x)] -  \mathbb{E}_{x \sim \mathbb{P}_\theta } [f(x)]\tag{2}
$$

ここで，$\mathbb{P}_\theta$ はGeneratorの出力分布（つまり，サンプルの生成分布）を表す．ここで $f$ は1-Lipschitz 関数である．しかし，式(2)の$\| f \|_L \leq 1$ を $\| f \|_L \leq K$ に置き換えたとしても，結局$K \cdot  W(\mathbb{P}_r, \mathbb{P}_{\theta})$ を考えることになる．よって，もしパラーメタ族

## 4. どうやって有効だと検証した？



## 5. 議論はある？



## 6. 次に読むべき論文は？



## 参考文献

Generative Adversarial Networks : Goodfellow  https://arxiv.org/abs/1406.2661

[From GAN to WGAN](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)