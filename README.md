# Implementation of HiFANet

1. Paper: Yuhang He, Lin Chen, Junkun Xie, Long Chen, Learning 3D Semantics from Pose-Noisy 2D Images with Hierarchical Full Attention Network.
[arXiv](https://arxiv.org/abs/2204.08084) 

2. Idea Summary: We propose to infer 3D point cloud semantics by aggregating 2D image semantics from temporal sequential observations. We consider LiDAR-camera pose
error which is common in real scenario by proposing the HiFANet. HiFANet is pose-noise tolerant and light-weight. It hierarchically aggregates
patch-level, instance-level and point-level semantics.

3. Implementation with Pytorch. After preparing the training dataset (see paper for details), run `main.py` to train the model.

4. If you find our work helpful, please cite as:  

```
@misc{HiFANet,
  doi = {10.48550/ARXIV.2204.08084},
  url = {https://arxiv.org/abs/2204.08084},
  author = {He, Yuhang and Chen, Lin and Xie, Junkun and Chen, Long},
  title = {Learning 3D Semantics from Pose-Noisy 2D Images with Hierarchical Full Attention Network},
  year = {2022},
}
```
