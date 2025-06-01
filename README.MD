[Repo Under Construction]

# SMPL Olympics 

[[paper]](https://arxiv.org/abs/2407.00187) [[website]](https://smplolympics.github.io/SMPLOlympics/)

<div float="center">
  <img src="assets/smplolympics.gif" />
</div>

## News 🚩

[July 5, 2024] Initial code release. Code trainable.


## TODOs

- [ ] Release trained models.

- [ ] Complete instructions. 

- [x] Release training data.

- [x] Release training code. 


## Intallation 

1. Create new conda environment and install pytroch:

```
conda create -n isaac python=3.8
[install pytorch]
pip install -r requirement.txt
```

2. Install [isaacgym](https://docs.nvidia.com/isaac/isaacgym/doc/setup.html)

3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. For SMPLX, please download the v1.1 version. Rename The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl

```

4. Download data and pretrained models with 
```
bash download_data.sh
```

# Commands

For each sport, we provide bash scripts to train baselines models (PPO/AMP/PULSE/PULSE+AMP). All scripts are in the `scripts` folder. Please check the contents of the script and pick one command (sometimes out of four) for training. 

To evaluate, append `no_virtual_display=True epoch=-1 test=True env.num_envs=1  headless=False ` to the end of the command. 


## Asset Attribution 
The soccer goalpost asset comes from: https://sketchfab.com/3d-models/football-goal-post-364cf6da76854862bfb77e650a80bd29
The tennis net asset comes from: https://sketchfab.com/3d-models/tennis-court-02fae7583fb447a484ee5b7c76bef0e6
The basketball hoop comes from: https://sketchfab.com/3d-models/canasta-baloncesto-basketball-hoop-bbef0dc4137b406f91709a692b338a3b

## Citation
If you find this work useful for your research, please cite our paper:
```
@article{luo2024smplolympics,
  title={Smplolympics: Sports environments for physically simulated humanoids},
  author={Luo, Zhengyi and Wang, Jiashun and Liu, Kangni and Zhang, Haotian and Tessler, Chen and Wang, Jingbo and Yuan, Ye and Cao, Jinkun and Lin, Zihui and Wang, Fengyi and others},
  journal={arXiv preprint arXiv:2407.00187},
  year={2024}
}       
```

Also, the multi-agent environment is built upon the following prior work:

```
@inproceedings{wang2024strategy,
  title={Strategy and skill learning for physics-based table tennis animation},
  author={Wang, Jiashun and Hodgins, Jessica and Won, Jungdam},
  booktitle={ACM SIGGRAPH 2024 Conference Papers},
  pages={1--11},
  year={2024}
}
```

