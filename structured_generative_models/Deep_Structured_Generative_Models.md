# Deep Structured Generative Models （07/10/2018 Kun Xu et.al）

# どんなもの？

現在のほとんどのGenerative Model は画像の構造（空間配置や物体間の意味的関係を含む）を説明する能力が欠落しているため，GANを発展させ，構造化情報をもつGenerative Model （特に配置やシーンの構造はsAOG（stochastic and-or graph）によって符号化される）を提案．そして，そのモデルはシーンや複雑なシーンの生成画像の本質的な構造を獲得でき，Detection Network は画像から場面の構造を推測することができる．実験結果は本質的な構造の模倣・本物のような画像生成の二つの観点から有効性を示した．

# 先行研究と比べてどこがすごい？

## 先行研究（対象研究）

- [Conditional GAN](https://arxiv.org/abs/1411.1784) ではone-hot vector をcodeに結合することで操作（物体を動かしたり，位置を変化させたりなど）を獲得していた．しかし，この方法では複雑で現実的な画像の関係のモデル化したとは言い難い．

- 対して Image grammear Model ( [Zhu et a., 2007](http://www.stat.ucla.edu/~sczhu/papers/Reprint_Grammar.pdf), [Zhao & Zhu](https://papers.nips.cc/paper/4236-image-parsing-with-stochastic-scene-grammar.pdf) ) では画像の構造をモデル化する理にかなった方法を提供する．しかしながら，Grammar Model は表現力が乏しい．つまり，各物体の複雑な外観をモデル化できない．

## 提案された手法

提案された新しいフレームワークではGenerative Model と Grammar Model の利点をいいとこ取りしたもの．

# 技術の手法のキモはどこ？

Grammar Model を用いて物体の配置をモデル化し，Generative Model は generating texture （色や外形・輪郭に応じた他のprimitives）によって，構造情報と visual primitives （線など）の橋渡しをするために使われる．さらに，Detection Model を適用し，現在の画像に対して最も可能性が高く，有効な configuration を見つけることで画像から隠れ構造を推測する．

**Fig1.** に提案手法のイラストが表されている．



# どうやって有効だと検証した？

# 議論はある？

# 次に読むべき論文は？