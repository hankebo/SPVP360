# SPVP360
# SPVP360

Spherical Convolution empowered FoV Prediction in360-degree Video Multicast with Limited FoV Feedbac

# abstract

Field of view (FoV) prediction is critical in 360-degree video multicast, which is a key component of the emerg-ing Virtual Reality (VR) and Augmented Reality (AR) applications. Most of the current prediction methodscombining saliency detection and FoV information neither take into account that the distortion of projected360-degree videos can invalidate the weight sharing of traditional convolutional networks, nor do they ade-quately consider the difficulty of obtaining complete multi-user FoV information, which degrades the predic-tion performance. This paper proposes a spherical convolution-empowered FoV prediction method, which isa multi-source prediction framework combining salient features extracted from 360-degree video with limitedFoV feedback information. A spherical convolution neural network (CNN) is used instead of a traditional two-dimensional CNN to eliminate the problem of weight sharing failure caused by video projection distortion.Specifically, salient spatial-temporal features are extracted through a spherical convolution-based saliency de-tection model, after which the limited feedback FoV information is represented as a time-series model basedon a spherical convolution-empowered gated recurrent unit network. Finally, the extracted salient video fea-tures are combined to predict future user FoVs. The experimental results show that the performance of theproposed method is better than other prediction method
Requirements


Citing

  @ARTICLE{9537928,
  author={Liu, Zhi and Li, Qiyue and Chen, Xianfu and Wu, Celimuge and Ishihara, Susumu and Li, Jie and Ji, Yusheng},
  journal={IEEE Network}, 
  title={Point Cloud Video Streaming: Challenges and Solutions}, 
  year={2021},
  volume={35},
  number={5},
  pages={202-209},
  doi={10.1109/MNET.101.2000364}}


Usage


This model is implemented by pytorch-gpu 0.3.1, and the detail of our computational environment is listed in 'requirement.txt'. Just run 'Test.py' to see the saliency prediction results on a test video.
