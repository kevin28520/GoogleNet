# GoogleNet
![GoogleNet](https://github.com/kevin28520/GoogleNet/blob/master/images/GoogleNet.png?raw=true)
1. the goal is to implement/optimize/use GoogleNet.
2. paper link: https://arxiv.org/pdf/1409.4842.pdf.
## Key points of GoogleNet
1. The architecture of GoogleNet is fantastic! I like the design so much, as it is mentioned in the original paper, the architecture is *carefully crafted design*. I think there are three main reasons for this design:

    1. because it makes sense and it works well (will come back to this point later).
    2. because of the limit of computational resources. By reading the paper, we can see that the authors considered a lot about the time complexity, the network size, the amount of parameters and training issue. And they figured out a way to `reduce data dimensionality by introducing a 1x1 convolution` before before big kernel sizes such as 3x3 conv and 5x5 conv.
    3. They added an `auxiliary classifier head` in the network, the reason being that when the network goes deeper and deeper, it becomes harder to train the network and it is hard for the network to converge. Regarding this issue, lots of methods in recent papers have been investigated to address the `gradient vanishing issue`. The auxiliary classifier is `disabled in the inference stage`, but it is enabled during training and it is very important for training. Because the loss in GoogleNet comes from two parts: one is from the final output layer, the second one comes from this auxiliary classifier head (and this loss is `weighted by 0.3`). Because its contribution of loss, `it helps increasing the gradient signal which propagates backwards`.
 2. Why does the architecture make sense?
 
    1. Perhaps the authors tested GoogleNet on lots of datasets, but the most import one is the `ImageNet dataset` which contains image classification and object localization. These are `Computer Vision tasks`, when dealing with Computer Vision, `convolutional operations` make sense since a Conv op can essentially `capture features from local receptive field`. For Computer Vision tasks, `scale matters`. Some papers use 3x3 or 5x5 in one layer, BUT `why don’t we use multiple scales of Conv op in the same layer` and concatenate the outputs together? And it makes great sense to do so because it can captures multiple scales of features. Basically the network can `grabs more information from the input`. And here are the original design of inception module (left) and the optimized one (right):
    ![](https://github.com/kevin28520/GoogleNet/blob/master/images/inception_module.JPG?raw=true)
    2. Not surprisingly, keeping stacking this kind of inception module leads to a `computational blow up`. This issue is solved by `inserting a 1x1 Conv before expensive Convs`. The functionality of 1x1 conv is to reduce data dimensionality from previous layer, it does so by `aggregating information` from a relatively higher dimensional tensor say 192 (the last dimension) to a relatively lower dimensional tensor say 96.
## TODO
