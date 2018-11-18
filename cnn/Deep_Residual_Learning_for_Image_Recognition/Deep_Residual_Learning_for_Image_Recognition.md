# Deep Residual Learning for Image Recognition

# Info

- Data : 12/ 2015
- Authors : Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- Journal reference: CVPR 2016

# どんなもの？



# ResNet の種類

本論文では 層数の違うResNet （ResNet-18/34/50/101/152）が提案されている．

## Residual Module について

ResNet-18/34 と ResNet-50/101/152 でResidual Block が異なる．

![fig5](/Users/yohei/Documents/papers/cnn/Deep_Residual_Learning_for_Image_Recognition/figures/fig5.png)

ResNet-18/34 は Fig.5 の左図のように3 × 3 の畳み込み層が 2 つ配置されている．正確には，畳み込み層に加えて，後述する batch normalization と ReLU が配置されており，本論文の ResNetでは下記のような構造の residual module が利用される:

<div align="center">
conv3x3 - bn - relu - conv3x3 - bn - add - relu
</div>

ここで， bn は batch normalization を，add は $F(x) $ と$x $の和をそれぞれ示している.この residual module の構造に関しては複数の改良手法が提案されている．

ResNet-50/101/152 のresidual moduleのはFig 5. の右図のような bottleneck バージョンと呼ばれるものを使用する．1 × 1 の畳み込みにより，次元削減を行った後に 3×3 の畳み込みを行い，その後さらに 1 × 1 により次元を復元するという形を取ることで，Fig 5. の左図と同等の計算量を保ちながら，より深いモデルを構築することができる．実際に，Fig 5. の左図の residual moduleを利用した ResNet-34 と比較して,同等のパラメータ数を持つFig 5. の右図のmoduleを利用した ResNet-50 は大きく精度が改善していることが報告されている．また実際に使われている residual moduleのbottleneck は以下のような構造である：

<div align="center">
conv1x1 - bn - relu - conv3x3 - bn - relu - conv1x1 - bn - add - relu
</div>

## 各ResNet の module 数について
ResNet-18/34/50/101/152 は，全て４個のResidual Block で構成されており，各の Residual Block の module 数についてまとめる．

residual 部分でdownsampling する回数は計４回で，サイズとchannel 数は

- ResNet-18/34

<div align="center">入力：56x56:64 -> 56x56:64 -> 28x28:128 -> 14x14:256 -> 7x7:512-> 出力 </div>

- ResNet-50/101/152
<div align="center">入力：56x56:64 -> 56x56:256 -> 28x28:512 -> 14x14:1024 -> 7x7:2048-> 出力 </div>

である．全てのResNetは224x224 の画像を入力サイズとするが，residual block series に行く前に

<div align="center">
conv(kernel_size: 7, stride: 2, padding: 3) - bn - relu - maxpooling(kernel_size: 3, stride: 2, padding: 1)
</div>
を適用するため，初めのredisual block に入力されるサイズは56x56である．また，ResNet-18/34 の通常のresidual module（Fig5. の右図）の場合，初めの residual blockはchannel 数を２倍にせず，size も1/2倍しない．しかし，ResNet-50/101/152 のbottleneck バージョン（Fig5. の左図）は始めの 大きなblock でchannel 数を4倍（64->256へ）し，size は1/2倍しないので <span style="color:red">要注意</span>（その後の大きなblock は通常通りのdownsampling をしている）．

さらに，residual block series の後はどのResNetも以下の構造となる．

<div align="center">GAP : average_pooling(kernel_size:7x7, stride: 1, padding: 0) 
<br> -> fc (in_fatures: 512 or 2048, out_features: 1000)：出力</div>

最後の fc 層の in_fatures はResNet-18/34 は 512であり，ResNet-50/101/152 は2048 であることを表している．

以下にResNet-18/34/50/101/152 それぞれの 大きなBlock ごとのBlock 数を示す．[a, b, c, d] はdownsampling（feature map sizeを1/2倍，channel 数を２倍） を行う residual moduleのつながりを Residual Block としてみた時の 各Block 内のresidual moodle の数である．つまり，a: 始めのResidual Blockを構成するresidual module の数を表す．b: ２つ目のResidual Blockを構成するresidual module の数を表す．

- ResNet-18：通常のResidual Module（Fig5. の右図）
<div align="center">
[2, 2, 2, 2]
</div>
- ResNet-34：通常のResidual Module（Fig5. の右図）

<div align="center">
[3, 4, 6, 3]
</div>

- ResNet-50：Bottleneck Residual Module（Fig5. の左図）

<div align="center">
[3, 4, 6, 3]
</div>

- ResNet-101：Bottleneck Residual Module（Fig5. の左図）

<div align="center">
[3, 4, 23, 3]
</div>

- ResNet-152：Bottleneck Residual Module（Fig5. の左図）

<div align="center">
[3, 8, 36, 3]
</div>



## Bottleneck Residual Block の実装について

Bottleneck Residual Module の場合，

<div align="center">
conv1x1 - bn - relu - conv3x3 - bn - relu - conv1x1 - bn - add - relu
</div>

初めのconv1x1のその際，一度channel 数を入力数より小さくし，次元削減を行う．その後，最後のconv1x1で次元を復元（Residual Block内で最後のBottleneck Residual Moduleならばchannel 数 を2倍に）する．以下にResidual Block ごとにchannel 数をいくつに削減するか（その数をmiddle channel とする）の詳細を記す．

- ResNet-50/101/152 全て，

<div align="center">
Residual Block(in_channels:64, middle_chainnels:64, out_channels:256) <br> - Residual Block(in_channels:256, middle_chainnels:128, out_channels:512)  <br>- Residual Block(in_channels:512, middle_chainnels:256, out_channels:1024) <br>- Residual Block(in_channels:1024, middle_chainnels:512, out_channels:2048) 
</div>



## Residual Block について

Residual Block の詳細を以下にResNet-13/34 と ResNet-50/101/52に分けて示す．

- ResNet-13/34

<div align="left">
Residual Block(in_channels, out_channels) = 
    <div align="center">
        Residual Module(in_channels,  out_channels) <br>
        - Residual Module(out_channels, out_channels) <br>
        - Residual Module(out_channels, out_channels) - .... <br>
        - Residual Module(out_channels,  out_channels)
    </div>
</div>

Residual Module：
<div align="left">
Residual Module(in_channels, out_channels)  = <br>
	<div align="center">
	conv3x3(in_channels: in_channels, out_channels: out_chainnels, stride: <span style="color:red">1 or 2</span>, padding: 1) - bn - relu<br>
        - conv3x3(in_channels: out_chainnels, out_channels: middle_chainnels, stride: 1 , padding: 1) - bn - add - relu
	</div>
</div>

Downsampling を行う Residual Module（つまりResidual Block内で一番初めのModule）は conv3x3 のstrideを<span style="color:red">2</span> にしてfeature map を1/2倍にする．



- ResNet-50/101/152

<div align="left">
Residual Block(in_channels, middle_chainnels, out_channels) = 
    <div align="center">
        Bottleneck Residual Module(in_channels, middle_chainnels, out_channels) <br>
        - Bottleneck Residual Module(out_channels, middle_chainnels, out_channels) <br>
        - Bottleneck Residual Module(out_channels, middle_chainnels, out_channels) - .... <br>
        - Bottleneck Residual Module(out_channels, middle_chainnels, out_channels)
    </div>
</div>
Bottlececk Residual Module: Facebook ResNet（後述）の場合．

<div align="left">
Bottleneck Residual Module(in_channels, middle_chainnels, out_channels)  = <br>
	<div align="center">
	conv1x1(in_channels: in_channels, out_channels: middle_chainnels, stride: 1, padding: 0) - bn - relu<br>
        - conv3x3(in_channels: middle_chainnels, out_channels: middle_chainnels, stride: <span style="color:red">1 or 2</span>, padding: 1) - bn - relu<br>
        - conv1x1(in_channels: middle_chainnels, out_channels: out_channels, stride: 1, padding: 0)  - bn - add - relu
	</div>
</div>
Downsampling を行うBottlececk Residual Module（つまりResidual Block内で一番初めのModule）は conv3x3 のstrideを<span style="color:red">2</span> にしてfeature map を1/2倍にする．



# 各ライブラリ（PyTorch, Chainer）のResNet の実装方法

## Residual Block について

PyTorch, Chainer 共に本論文で提案された構成で residual block を構築している．


## shortcut の downsampling 方法について

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

まず，bottleneck（fig5 の右図） を用いて構築される ResNet-50/101/152 にはdownsampling（こちらはshortcut ではなくストレートにResidual Block を下る時に行うdownsampling 操作，つまりfeature map を1/2 にする操作） を行うConvolutionを最初の1x1 で行うのか，2層目の 3x3 で行うのかの２種類がある．Chainer はそのどちらも実装済みであるが，PyTorch はFacebook ResNet の手法のみ実装されている．
1. original MSRA ResNet ：kernel_size: 1x1 のConvolution でdownsampling（提案された論文に書かれてあった手法）．
2. Facebook ResNet：kernel_size：3x3 のConvolution でdownsampling．

また，Chainer のResNet は ResNet-50/101/152 しか実装されていないので，Residual Block がbottleneck のものしか実装されていない．よって以下は bottleneck のものについて説明する．

MSRA ResNet も Facebook ResNet も shortcut のdownsampling 操作は以下のよう．

- Convolution (kernel_size: 1x1, strides: 2, padding: 0, out_channel : Residual Blockの出力のchannel size) の後，BatchNormalizationを適用．この操作を適用したものをResidual Blockの出力と add し，最後に ReLU を適用し最終的な出力とする．（つまりPytorch と同じ．）

downsampling : https://github.com/chainer/chainer/blob/master/chainer/links/model/vision/resnet.py#L633-L636

add : https://github.com/chainer/chainer/blob/master/chainer/links/model/vision/resnet.py#L642-L643



## 結論

PyTorch も Chainer もshortcut の downsampling 操作は，

- Convolution (kernel_size: 1x1, strides: 2, padding: 0, out_channel : Residual Blockの出力のchannel size) の後，BatchNormalizationを適用．この操作を適用したものをResidual Blockの出力と add し，最後に ReLU を適用し，最終的な出力とする．

である．

