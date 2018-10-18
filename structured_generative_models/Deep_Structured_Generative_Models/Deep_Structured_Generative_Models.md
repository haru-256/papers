# Deep Structured Generative Models （07/10/2018 Kun Xu et.al）

# どんなもの？

現在のほとんどのGenerative Model は画像の構造（空間配置や物体間の意味的関係を含む）を説明する能力が欠落しているため，GANを発展させ，構造化情報をもつGenerative Model （特に配置やシーンの構造はsAOG（stochastic and-or graph）によって符号化される）を提案．そして，そのモデルはシーンや複雑なシーンの生成画像の本質的な構造を獲得でき，Detection Network は画像から場面の構造を推測することができる．実験結果は本質的な構造の模倣・本物のような画像生成の二つの観点から有効性を示した．

# 先行研究と比べてどこがすごい？

## 先行研究（対象研究）

- [Conditional GAN](https://arxiv.org/abs/1411.1784) ではone-hot vector をcodeに結合することで操作（MNISTならば，生成したい数字を指定するなど）を獲得していた．しかし，この方法では複雑で現実的な画像の関係のモデル化したとは言い難い．

- 対して Image grammear Model ( [Zhu et a., 2007](http://www.stat.ucla.edu/~sczhu/papers/Reprint_Grammar.pdf), [Zhao & Zhu](https://papers.nips.cc/paper/4236-image-parsing-with-stochastic-scene-grammar.pdf) ) では画像の構造をモデル化する点では理にかなった方法である．しかしながら，Grammar Model は表現力が乏しく，各物体の複雑な外観をモデル化できない．

## 提案された手法

提案された新しいフレームワークではGenerative Model と Grammar Model の利点をいいとこ取りしたもの．

画像の生成はGenerative Modelに任せることでsAOGは pixelwise image だけでなく，高解像度のシーンのレイアウトを生成する必要がなくなった．

# 技術の手法のキモはどこ？

Grammar Model を用いて物体の配置をモデル化し，Generative Model は構造情報と visual primitives （線など）の橋渡しをするために使われる．さらに，Detection Model を適用し，現在の画像に対して最も可能性が高く，有効な configuration を見つけることで画像から隠れ構造を推測する．

- sAOG: 構文木でシーンが表されている．パラメータは最尤推定によって学習される．

Fig1., Fig2.  に提案手法のイラストが表されている．

<img src="/Users/yohei/Documents/papers/structured_generative_models/Deep_Structured_Generative_Models/figures/fig3.png" width=40% align="middle">

<img src="/Users/yohei/Documents/papers/structured_generative_models/Deep_Structured_Generative_Models/figures/fig2.png" width=100% align="middle">

Fig2. A) に示すように本研究でのモデルは潜在変数モデルであり，同時分布は以下のように定式化される．

$$
P(X, G) =P_G (G) P_{X|G} (X| G)
$$
ここで，$X$ は画像データを表し，$ G$ は画像自体の構造を意味する．$P_G$ は構造の確率関数であり，$P_{X|G}$ 構造が与えられたもとでの画像の確率を表す．このモデルでは潜在変数 $G$ は sAOG によって定義され，そのサンプルは構文木 $g$ である．条件付き分布$P_{X|G}$は，

$$
P_{X|G} = P_{X|M}(X|m(G))
$$
とし，$m(\cdot)$ は構造 $G$ を2-D bounding box へと変換する関数である．そして$P_{X|M}$ は bounding box map をimage-to-image model を用いてrealistic image へと変換する．

Stochastic And-Or Graph (sAOG) は3次元的配置と物体間の関係をモデル化するImage Grammer Model として使われる．

## Grammar Model (sAOG)

### 詳細

Stochastic image grammar は画像の構造をモデル化する（通常，sAOGとして定式化される）確率的フレームワークである．sAOGは < S, V, R, E, P > の５つで定義される．

1. S: 全体のシーンを表すルートノード
2. V: 有限頂点集合, オブジェクトを表す終端ノード$V_T$ やより高レベル（抽象的）な構造を洗わす$V_N$ を含む．
3. R: 生成規則の集合, 要素は特定の$V_N$ を$V$ の子ノードの列に分解する．
4. $E \subset V \times V$ : ノード間の関係の集合．それぞれの関係のタイプは and-or graph のノード間のエッジのタイプで表される．
5. P: 全ての有効な構文木（grammar model から生成されたグラフ）についての確率を定義する確率関数．

sAOG はグラフで表現されるシーンの構造を学習し，その構造を，オブジェクトを射影することでbounding box へ変換する．



Fig 3. に本論文で扱ったを図式化した． 

<img src="/Users/yohei/Documents/papers/structured_generative_models/Deep_Structured_Generative_Models/figures/fig1.png" width=60% align="middle">

scene structure $g$ は parse graph として表される．scene node $S$ は初めに scene configure を選ぶ．scene configure は[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/)  でいうとscene内のオブジェクトの数を表す．scene configure は $n_s$個（sceneないの物体の数）のオブジェクトノード ${o_i}_{i=1}^{n_s}$ とカメラノード $c$ を子にもつ．物体間の関係（左にある，前にある）はノード間のエッジで表される． そして，各オブジェクトノードはそのオブジェクトの視覚的属性（色や形，物体の位置）を意味するinstance を子ノードにもつ．

sAOGの構文木は以下のように定式化される．

$$
\begin{align}
P_G(g) &= P_s(Ch_s) \prod_{o \in O_g} P_{o} (Ch_o) \times P_{\varepsilon}(g) \tag 1\\
P_{\varepsilon}(g) &= \frac{1}{z} \exp(-\varepsilon(g))
\end{align}
$$

ここで$P_S$ はscene configuration の確率であり，$P_o$ は あるオブジェクト $o$ についての object instance の確率を表す．そして$P_{\varepsilon}(g)$ は関係 $E_{g}$ と他の制約の上で定義された energy based distribution である．

$$
\varepsilon(g)  = \lambda_d \sum_{e \in E_{g}} \varepsilon_d (e) + \lambda_c \sum_{o \in O_{g}} \varepsilon_c (o) + \lambda_h \sum_{o \in O_{g}} \varepsilon_h (o)
$$
$\varepsilon_d (e), \varepsilon_c (o), \varepsilon_h (o)$ の意味は論文の式(3) の下 を参照．

つまり，式(1) でのパラメータは $P_s, P_o, \lambda = [\lambda_d, \lambda_c, \lambda_h]$ である．

### 学習

パラメータ$P_s, P_o, \lambda = [\lambda_d, \lambda_c, \lambda_h]$ の学習は学習データより，以下のように最尤推定で行う．

$$
\max_{P_s, P_o, \lambda}  \sum_{g \in \mathcal{D}_g} \log P_G(g)
$$

$\lambda$ はcontractive divergence を使用し学習する．具体的なパラメータの更新ルールは，論文中の式(6), (7)で与えられる．

### Infere Grammar from image

画像から構造を推定するために，本研究では物体検出モデルを用いる．物体検出により3D location や 他の視覚的属性を求める．属性を持つオブジェクト集合$O$ が与えらた時，そのオブジェクト間の関係 $E^{*}​$は以下のようにMAP推定で求められる．

$$
E^{*} = \underset{E}{\operatorname{argmax}} P_G (E|O)
$$

$E$は関係の集合，$O$ はオブジェクトの集合，そして$P_G(E|O)​$  は object instanceを与えられた元での関係の事後分布を意味する．MAP 推定を求めるため，MCMCが使われる．

## Pix2pix Model 

本論文ではU-netを改造している．U-netでは，情報の保持を目的としてDecoder と Encoder の同一レベルblock間にskip connection があった．本研究では物体を生成するだけでなく，bounding box をより適切な物体の形に変換することを必要とする．よって，U-net の高レベル層では skip connection を取り除いた．



# どうやって有効だと検証した？

画像の生成，推論ともに，[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) データセットを用いて実験した．

## 実験設定

### CLEVRデータセット

- 学習データ数: 70000
- 解像度: 480x320
- 各画像はキューブや球，シリンダーが写っており，それぞれ２種類の材質，８種類の色がありえる．計48種類の物体が写っている．

> The CLEVR dataset consists of 70000 training images with resolution 480×320, each consists of several simple objects such as cube, sphere and cylinder with 8 colors and 2 materials, totally 48 kinds of object labels. The 3D locations and spatial relations are given according to the location of the camera which is shared over the whole dataset. We use the provided attributes and 3D relations to build the sAOG, and then feed the projected bounding box images to refinement network for the realistic images

データの中身等の詳しいことは`tortoise7`の`~/Documents/tlab/data_CLEVR/CLEVR_v1_scenes.ipynb`へ

### Detection Model

- Detection (recognition) ModelとしてResNet-52構造のFaster-RCNNを使用．
- 各物体の3次元配置の回帰を行うために，中間層1層で64の隠れ素子を持つMLPを使用．

### Structure of GAN 

__Generator__
- EncoderとDecoderがそれぞれ8のconvolutio blobck と deconvolution block を持つ [U-net](https://arxiv.org/abs/1505.04597) を使用．
- shape transformation を可能とするために U-net の最高位2つのレベルの層には skip connect をつけない．
- U-net は入力として9 channel のbounding box を入力される．３つは物体の48種類のラベルを表す．残りの６つは 0◦ ∼ 15◦, 15◦ ∼ 30◦ … 75◦ ∼ 90◦と離散化された回転の one-hot vector である．
- まず入力画像は 256x256 にresizeされ，出力された画像を再度 480x320 へresizeする.

__Discriminator__
- 6-layer convolution network

__Loss__
Loss はPix2Pix と同じく，以下の２項のロスを用いた．

- 通常のGAN Loss
$$
L_{cGAN} (G, D) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x, z}\left[\log \left(1-D\left(x, G(x, z)\right)\right)\right]
$$
- L1 loss
$$
L_{L1} (G) = \mathbb{E}_{x, y, z}\left[\| y-G(x, z)\|_1 \right]
$$
##画像の生成からの観点

Fig 4. に示すようにVAE, [SN-GAN](https://arxiv.org/abs/1802.05957)と比較しても，提案手法は構造を持った画像を生成することができている．Fig 4.の E)に示すように Inference Network を用いると画像を構造情報まで落とし込むことができ，refinement network によって再び画像へと戻すことができる．これを利用すると480x320の画像を1KBに圧縮することができる．（もしくは，1KBで画像を圧縮することができる？）

<img src="/Users/yohei/Documents/papers/structured_generative_models/Deep_Structured_Generative_Models/figures/fig4.png" width=100% align="middle">

## 条件付き生成からの観点
提案手法には Image Grammar Model が埋め込まれているため，ある構造の条件を持った画像を生成することができる．Fig 5. では上下に5枚ずつ画像が並べられているが，これらは，各行で同じ属性（色，形）と関係を持たせた符号を用いて生成した画像である．

<img src="/Users/yohei/Documents/papers/structured_generative_models/Deep_Structured_Generative_Models/figures/fig5_6.png" width=100% align="middle">


# 議論はある？

実装の再現が現在（10/17/2018）ではできなさそう．コードがない，モデルの詳細（初期値や活性化関数等）がない，Image Grammar Model の理解が乏しい．

# 次に読むべき論文は？

1. CARL DOERSCH, Tutorial on Variational Autoencoders : https://arxiv.org/pdf/1606.05908.pdf
   GANとは異なる生成モデルについて知るため．
2. KINGMA, Glow: Generative Flow with Invertible 1x1 Convolutions : https://arxiv.org/abs/1807.03039
   リバーシブル（画像からコードが得られる）なモデルについて知っておきたいため．
3. GULRAJANI, Improved Training of Wasserstein GANs : https://arxiv.org/abs/1704.00028
   WGAN の続きとして．
4. SONG-CHUN ZHU and DAVID MUMFORD, A Stochastic Grammar of Images : http://www.stat.ucla.edu/~sczhu/papers/Reprint_Grammar.pdf
   Stochastic Image Grammar モデルについて
5. Image Generation from Scene Graphs : https://arxiv.org/abs/1804.01622
   本論文とは違う視点（scene graphを条件とした構造つきGenerative Model） の論文．graph convolution を使用．
6. PHILIPISOLA et al. Image-to-Image Translation with Conditional Adversarial Networks: https://arxiv.org/abs/1611.07004
   pix2pix
7. TING-CHUN WANG et al. High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs: https://arxiv.org/abs/1711.11585
   pix2pixHD
