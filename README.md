# SDA-GAN
## [Semi-Supervised Domain Alignment Learning for Single Image Dehazing](https://ieeexplore.ieee.org/document/9965586)      (IEEE TRANSACTIONS ON CYBERNETICS'22)
Yu Dong, Yunan Li, Qian Dong, He Zhang, Shifeng Chen



<center >
    <img src= "https://github.com/WeilanAnnn/SDA-GAN/blob/master/Fig/network.png"/>
</center>


  In this article, a semi-supervised learning paradigm is proposed for image dehazing. This paradigm has one branch with synthesized hazy images for supervised learning and the other with real hazy images for unsupervised learning. For a better generalization on real-world images, a domain alignment strategy is adapted to shorten the distance between the high-level haze features of the two branches in the latent space rather than directly share the network weights. Moreover, we introduce a haze-aware attention module according to the local entropy theory to facilitate adaptive attention on hazy regions. Therefore, our semi-supervised design can powerfully alleviate the domain-shift problem and generate better-dehazed results with clearer details and more natural color. Extensive experiments have demonstrated that our method performs favorably against state-of-the-art methods on both synthetic datasets and real-world hazy images.

<center >
    <img src="https://github.com/WeilanAnnn/SDA-GAN/blob/main/Fig/Comparisons.png" width="1000"/>
</center>


# Citation
@ARTICLE{9965586,
  author={Dong, Yu and Li, Yunan and Dong, Qian and Zhang, He and Chen, Shifeng},
  journal={IEEE Transactions on Cybernetics}, 
  title={Semi-Supervised Domain Alignment Learning for Single Image Dehazing}, 
  year={2022},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TCYB.2022.3221544}}

# Contact
The code is a little bit messy and we need more time to organize.

For algorithmic details, please refer to our paper. If you need test results of your images urgently, please contact yu.dong@siat.ac.cn.

