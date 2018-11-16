# 物体認識のための畳み込みニューラルネットワークの研究動向

# Info

- Date：11/2018
- Authors：Yusuke UCHIDA, Takayoshi, YAMASHITA
- Journal reference：電子情報通信学会論文誌

# どんなもの？

AlexNet 以降の代表的なCNN についてのサーベイと，代表的なモデルに対して複数のデータセットでの評価をまとめた論文．

# 代表的なCNNの変遷

## ILSVRC 2012 〜 2017 のCNN

### AlexNet：2012優勝
AlexNet に導入された主な重要な技術
- LRN : Local Response Normalization
- Overlapping Pooling
- Dropout

### ZFNet：2013優勝

AlexNet の重みの可視化を行い，以下の問題点２つを見つけ改善した．

1. 最初の畳み込み層のフィルタが，大きな サイズのカーネルを利用していることから極端な高周 波と低周波の情報を取得するフィルタとなっており， それらの間の周波数成分を取得するフィルタが殆ど無 かった．
2. 2 層目の特徴 マップにおいて，エイリアシングが発生していること．

解決するためにそれぞれ以下の改善をした．

1. 最初の畳み込み層のフィル タサイズを11から7に縮小
2. strideを4から 2 に縮小する

### GoogleNet：2014年度優勝

GoogleNet に導入された主な重要な技術

- Inception モジュール
- Global Average Pooling （GAP）
- auxiliary loss

### VGG

### ResNet


## 最近のCNNについて



# 代表的なCNNの概要



# 代表的なCNNの評価



# 議論はある？



# 次に読むべき論文は？

- A. Krizhevsky, I. Sutskever, and G. E. Hinton, Im-agenet classication with deep convolutional neural networks, inProc. of NIPS , 2012.
  AlexNetの論文
- M. D. Zeiler and R. Fergus, Visualizing and under-standing convolutional networks, in Proc. of ECCV , 2014.
  ZFNet の論文

- 参考: 〜2017までのCNNについて：https://qiita.com/yu4u/items/7e93c454c9410c4b5427

