# High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs

# Info

- Data : 11/30/2017
- Authors : Ting-Chun Wang, Ming-Yu Liu, et al.
- Journal reference : CVPR 2018

# どんなもの？

新たなadversalial loss とmulti-scale generator, discriminator architectures を用いて semantic label map から高解像度（2048x1024）の画像を生成する手法を提案した．

さらに提案フレームワークを拡張させ，２つの機能を用いてinteractive な視覚的操作を可能にした．

1. object segmentation instance 情報を統合し，オブジェクト操作を可能にさせた．
2. ある１つの入力に対して多様な出力を生成させる手法を提案し，ユーザーがインタラクティブに物体の外観を変更することを可能にした．

# 先行研究と比べてどこがすごい？

主に比較対象とする先行研究は [Q. Chen](https://arxiv.org/abs/1707.09405) のperceptual loss を改良した手法．

その先行研究で，adversarial training はunstable であり，高解像度の画像生成タスクにおいて学習が失敗しやすいということが分かった．代わりに彼らはsynthesize images について改良した perceoutual loss を用いた．これは高解像度の画像を生成することができるが，しばしば物体の詳細やrealistic なtextures は欠落する．

本論文では上のSoTA の手法の主な問題点である以下の２つについて取り組んだ．

1. GANs を用いた高解像度の画像生成
2. 先行研究での生成された高解像度画像での，物体の詳細とrealistic texture の欠落

# 技術や手法のキモはどこ？



# どうやって有効だと検証した？



# 議論はある?



# 次に読むべき論文は?



## 先行研究

- Q. Chen and V. Koltun. Photographic image synthesis with cascaded refinement networks.
   In IEEE International Conference on Computer Vision (ICCV), 2017.