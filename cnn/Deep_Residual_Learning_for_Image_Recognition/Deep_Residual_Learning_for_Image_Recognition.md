# Deep Residual Learning for Image Recognition

# Info

- Data : 12/ 2015
- Authors : Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- Journal reference: CVPR 2016

# どんなもの？



# 各ライブラリ（Pytorch, Chainer）のResNet のdownsampling 実装方法

ResNet で shortcutのdownsampling 操作（feature map を1/2（stride:2の畳み込み）にし，channel をResidual Block の出力に合わせる操作） が必要なところ（下図の右での破線のshortcut）では各ライブラリでどのように実装されているのかをまとめる．

![fig3](/Users/yohei/Documents/papers/cnn/Deep_Residual_Learning_for_Image_Recognition/figures/fig3.png)

PyTorch ，Chainer 共に shortcut tyep は projection shortcuts  であり，identity shortcuts は実装されていない．

## PyTorch

PyTorch では shortcut の downsampling 操作は以下のよう．

- Convolution (kernel_size: 1x1, strides: 2, padding: 0, out_channel : Residual Blockの出力のchannel size) の後，BatchNormalizationを適用．この操作を適用したものをResidual Blockの出力と add し，最後に ReLU を適用し最終的な出力とする．

downsampling：https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L125-L129

add：https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L52-L56

## Chainer

![fig5](/Users/yohei/Documents/papers/cnn/Deep_Residual_Learning_for_Image_Recognition/figures/fig5.png)

まず，bottleneck（fig5 の右図） を用いて構築されるResNet にはdownsampling（こちらはshortcut ではなくストレートにResidual Block を下る時に行うdownsampling 操作，つまりfeature map を1/2 にする操作） を行うConvolutionを最初の1x1 で行うのか，2層目の 3x3 で行うのかの２種類がある．Chainer はそのどちらも実装済みであるが，PyTorch はoriginal のMSRA ResNet の手法のみ実装されている．
1. original MSRA ResNet ：kernel_size: 1x1 のConvolution でdownsampling（提案された論文に書かれてあった手法）．
2. Facebook ResNet：kernel_size：3x3 のConvolution でdownsampling．

また，Chainer のResNet は Residual Block がbottleneck のものしか実装されていない（11/18/2018 現在）．よって以下は bottleneck のものについて説明する．

MSRA ResNet も Facebook ResNet も shortcut のdownsampling 操作は以下のよう．

- Convolution (kernel_size: 1x1, strides: 2, padding: 0, out_channel : Residual Blockの出力のchannel size) の後，BatchNormalizationを適用．この操作を適用したものをResidual Blockの出力と add し，最後に ReLU を適用し最終的な出力とする．（つまりPytorch と同じ．）



downsampling : https://github.com/chainer/chainer/blob/master/chainer/links/model/vision/resnet.py#L633-L636

add : https://github.com/chainer/chainer/blob/master/chainer/links/model/vision/resnet.py#L642-L643



## 結論

PyTorch も Chainer もshortcut の downsampling 操作は，

- Convolution (kernel_size: 1x1, strides: 2, padding: 0, out_channel : Residual Blockの出力のchannel size) の後，BatchNormalizationを適用．この操作を適用したものをResidual Blockの出力と add し，最後に ReLU を適用し，最終的な出力とする．

である．

