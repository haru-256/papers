# Deep Structured Generative Models （07/10/2018 Kun Xu et.al）

# どんなもの？

現在のほとんどのGenerative Model は画像の構造（空間配置や物体間の意味的関係を含む）を説明する能力が欠落しているため，GANを発展させ，構造化情報をもつGenerative Model （特に配置やシーンの構造はsAOG（stochastic and-or graph）によって符号化される）を提案．そして，そのモデルはシーンや複雑なシーンの生成画像の本質的な構造を獲得でき，Detection Network は画像から場面の構造を推測することができる．実験結果は本質的な構造の模倣・本物のような画像生成の二つの観点から有効性を示した．

# 先行研究と比べてどこがすごい？

## 先行研究（対象研究）

- [Conditional GAN](https://arxiv.org/abs/1411.1784) ではone-hot vector をcodeに結合することで操作（物体を動かしたり，位置を変化させたりなど）を獲得していた．しかし，この方法では複雑で現実的な画像の関係のモデル化したとは言い難い．

- 対して Image grammear Model ( [Zhu et a., 2007](http://www.stat.ucla.edu/~sczhu/papers/Reprint_Grammar.pdf), [Zhao & Zhu](https://papers.nips.cc/paper/4236-image-parsing-with-stochastic-scene-grammar.pdf) ) では画像の構造をモデル化する理にかなった方法を提供する．しかしながら，Grammar Model は表現力が乏しい．つまり，各物体の複雑な外観をモデル化できない．

## 提案された手法

提案された新しいフレームワークではGenerative Model と Grammar Model の利点をいいとこ取りしたもの．

画像の生成はGenerative Modelに任せることでsAOGは pixelwise image だけでなく，高解像度のシーンのレイアウトを生成する必要がなくなった．

# 技術の手法のキモはどこ？

Grammar Model を用いて物体の配置をモデル化し，Generative Model は構造情報と visual primitives （線など）の橋渡しをするために使われる．さらに，Detection Model を適用し，現在の画像に対して最も可能性が高く，有効な configuration を見つけることで画像から隠れ構造を推測する．

- sAOG: 構文木でシーンが表されている．パラメータは最尤推定によって学習される．

**Fig1.** に提案手法のイラストが表されている．

Stochastic And-Or Graph (sAOG) は3次元的配置と物体間の関係をモデル化するImage Grammer Model として使われる．

## sAOG

Stochastic image grammar は画像の構造をモデル化する（通常，sAOGとして定式化される）確率的フレームワークである．

# どうやって有効だと検証した？

画像の生成，推論ともに，[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) データセットを用いて実験した．

## CLEVRデータセット

- 学習データ数: 70000
- 解像度: 480x320
- 各画像はキューブや球，シリンダーが写っており，それぞれ２種類の材質，８種類の色がありえる．計48種類の物体が写っている．

> The CLEVR dataset consists of 70000 training images with resolution 480×320, each consists of several simple objects such as cube, sphere and cylinder with 8 colors and 2 materials, totally 48 kinds of object labels. The 3D locations and spatial relations are given according to the location of the camera which is shared over the whole dataset. We use the provided attributes and 3D relations to build the sAOG, and then feed the projected bounding box images to refinement network for the realistic images

## Detection Model

- Detection (recognition) ModelとしてResNet-52構造のFaster-RCNNを使用．
- 各物体の3次元配置の回帰を行うために，中間層1層で64の隠れ素子を持つMLPを使用．

## Structure of GAN 

- Generator
  - EncoderとDecoderがそれぞれ8のconvolutio blobck と deconvolution block を持つ [U-net](https://arxiv.org/abs/1505.04597) を使用．
  - shape transformation を可能とするために U-net の最高位2つのレベルの層には skip connect をつけない．
  - U-net は入力として9 channel のbounding box を入力される．３つは物体の48種類のラベルを表す．残りの６つはs 0◦ ∼ 15◦, 15◦ ∼ 30◦ … 75◦ ∼ 90◦と離散化された回転の one-hot vector である．
  - まず入力画像は 256x256 にresizeされ，出力された画像を再度 480x320 へresizeする.
- Discriminator
  - 6-layer convolution network

##画像の生成からの観点
Fig 4. に示すようにVAE, [SN-GAN](https://arxiv.org/abs/1802.05957)と比較しても，提案手法は構造を持った画像を生成することができている．Fig 4.の E)に示すように Inference Network を用いると画像を構造情報まで落とし込むことができ，refinement network によって再び画像へと戻すことができる．これを利用すると480x320の画像を1KBに圧縮することができる．（もしくは，1KBを使用して画像を圧縮することができる？）
## 条件付き生成からの観点
提案手法には Image Grammar Model が埋め込まれているため，ある構造の条件を持った画像を生成することができる．Fig 5. では上下に5枚ずつ画像が並べられているが，これらは，各行で同じ属性（色，形）と関係を持たせた符号を用いて生成した画像である．



# 議論はある？

# 次に読むべき論文は？