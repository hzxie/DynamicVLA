<img src="https://www.infinitescript.com/projects/DynamicVLA/DynamicVLA-Logo.webp" height="100px" align="right">

# DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation

[Haozhe Xie](https://haozhexie.com), [Beichen Wen](https://wenbc21.github.io/), Jiarui Zheng, [Zhaoxi Chen](https://frozenburning.github.io/), [Fangzhou Hong](https://hongfz16.github.io/), [Haiwen Diao](https://paranioar.github.io/), [Ziwei Liu](https://liuziwei7.github.io/)

S-Lab, Nanyang Technological University

[![codefactor badge](https://www.codefactor.io/repository/github/hzxie/DynamicVLA/badge)](https://www.codefactor.io/repository/github/hzxie/DynamicVLA)
![Counter](https://api.infinitescript.com/badgen/count?name=hzxie/DynamicVLA)
[![arXiv](https://img.shields.io/badge/arXiv-2601.22153-b31b1b.svg)](https://arxiv.org/abs/2601.22153)
[![YouTube](https://img.shields.io/badge/Spotlight%20Video-%23FF0000.svg?logo=YouTube&logoColor=white)](https://youtu.be/NmJnHcI04_Q)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-DOM%20Dataset-orange)](https://huggingface.co/datasets/hzxie/DOM)

![Teaser](https://github.com/user-attachments/assets/ffc2071a-c4b8-4ebf-9a41-870de65bb3da)

> ⚠️ **To reproduce the paper, use Isaac Sim 4.5.0 + Isaac Lab 2.2.1.**
> The paper numbers (47.06%) were obtained on Isaac Sim 4.5.0 / Isaac Lab 2.2.1. On Isaac Sim 5.1.0 / Isaac Lab 2.3.2 the released checkpoint scores 40.67%, and the gap cannot be closed by configuration: PhysX 5.6 changed the contact behavior of cylinder colliders, so lying cans/cups/bottles roll and slip differently in 30 of the 90 test environments. Report 5.1 results as a separate setting; see [#22](https://github.com/hzxie/DynamicVLA/issues/22).

## Cite this work📝

```
@inproceedings{xie2026dynamicvla,
  title     = {{DynamicVLA:} A Vision-Language-Action Model for Dynamic Object Manipulation},
  author    = {Xie, Haozhe and
               Wen, Beichen and
               Zheng, Jiarui and
               Chen, Zhaoxi and
               Hong, Fangzhou and
               Diao, Haiwen and
               Liu, Ziwei},
  booktitle = {NeurIPS},
  year      = {2026}
}
```

## Dataset and Pretrained Models 🛢️

### DOM Dataset

- [DOM Training Set](https://huggingface.co/datasets/hzxie/DOM) – for training DynamicVLA
- [DOM Testing Set](https://gateway.infinitescript.com/?f=DOM-Test) – for benchmarking; includes test configurations and a subset of 3D scenes
- [DOM 3D Objects](https://gateway.infinitescript.com/?f=DOM-3D-Objects) – assets for data generation and benchmarking
- [DOM 3D Scenes](https://gateway.infinitescript.com/?f=DOM-3D-Scenes) – full scene assets for data generation

### Pretrained Models

- [DynamicVLA  (trained on DOM)](https://huggingface.co/hzxie/dynamic-vla-DOM)


## Installation 📥

We recommend using **conda** to create two separate environments:

- one for **model training & inference**
- one for **Isaac Lab simulation & evaluation**

### PyTorch Environment

- Install **Python 3.10** and **PyTorch 2.7.1** *(Other versions should work, but are not fully tested)*
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Isaac Lab Environment

- Install **Python 3.11** *(Other versions should work, but are not fully tested)*
- Install **Isaac Sim 4.5.0** and **Isaac Lab 2.2.1** (recommended, matches the paper) or **Isaac Sim 5.1.0** and **Isaac Lab 2.3.2** (see the note above)
  Follow the official guide: https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/index.html (or https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html for 5.1.0)
- Install additional dependencies:

```bash
pip install shapely pyzmq h5py
```

> ⚠️ **numpy compatibility:** Isaac Sim 5.1 pins `numpy==1.26`, and observations/actions are exchanged between the two environments as pickled numpy arrays over ZeroMQ. Keep `numpy==1.26.x` in the PyTorch environment as well (this is what `requirements.txt` pins). With numpy 2.x on the PyTorch side, the evaluation server crashes with `ModuleNotFoundError: No module named 'numpy._core'`.

## Benchmarking Your Policy 🏅

### Prepare scenes and objects

Download [DOM Testing Set](#dom-dataset) (including a subset of DOM 3D scenes) and [DOM 3D Objects](#dom-dataset).

```
PROJECT_ROOT/
├── objects/           # Put DOM 3D Objects here
├── scenes/            # Put DOM 3D Scenes here
|   └── textures       # Put the textures of 3D scenes here
|   └── *.usd          # Put the USD files of 3D scenes here
├── tests/             # Put DOM Testing Set here
|   └── *.json
|── test-envs.txt      # The list of test environments (included in DOM Testing Set)
├── datasets/          # Generated simulated datasets will be stored here
└── dynamic-vla/       # git clone https://github.com/hzxie/DynamicVLA dynamic-vla
    └── runs           # Create folder for evaluation and checkpoints output
```

### Run Policy Evaluation Server

> ⚠️ This step requires the **Isaac Lab environment**

From the `PROJECT_ROOT/dynamic-vla` directory, run:

```bash
python3 simulations/evaluate.py \
    --scene_dir ../scenes \
    --output_dir ../output/evaluation \
    --env_cfg ../test-envs.txt \
    --enable_cameras --headless -n 20 --save
```

**Arguments:**

- `test-envs.txt` are provided by [DOM Testing Set](#dom-dataset)
- `-n 20`: run 20 trials per environment
- `--save`: save evaluation videos to `output_dir`
- `--headless`: run without GUI
- `--enable_cameras`: enable visual observations

> ⚠️ **Multi-GPU machines:** run the simulation and the policy on two different GPUs (inference latency affects the results). Isaac Lab 2.3.2 only supports `cuda:0` for its Fabric views, so do **not** pass `--device cuda:N` with `N > 0`; select the GPU with `CUDA_VISIBLE_DEVICES` instead and keep the default `--device cuda:0`, e.g.
>
> ```bash
> CUDA_VISIBLE_DEVICES=1 python3 simulations/evaluate.py ... \
>     --kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=1"
> ```
>
> The `--kit_args` part keeps the renderer on the same physical GPU as well. Then start the policy with `CUDA_VISIBLE_DEVICES=0` (or any other free GPU).

### Run Policy Inference

> ⚠️ This step requires the **PyTorch environment**

From the `PROJECT_ROOT/dynamic-vla` directory, run:

```bash
python3 scripts/inference.py \
    -p /path/to/vla-checkpoint \
    -r euler -d -s
```

**Arguments:**

- `-p`: path to the trained model checkpoint
- `-r euler`: use Euler angles for rotation representation
- `-d`: enable **delta actions** *(actions are relative to current state)*
- `-s`: enable **contiguous inference** *(if supported by the model)*

## Simulated Dataset Generation 🧪

> ⚠️ This step requires the **Isaac Lab environment**

### IsaacSim Simulation

From the `PROJECT_ROOT/dynamic-vla` directory, you can generate synthetic data using:

```bash
python3 simulations/simulate.py \  
        --headless  --enable_cameras  --seed 42  --save  --task place
```

Example configuration file: `simulations/configs/sim_cfg.yaml`

**Arguments:**

-   `--task` : task type to simulate. Options: `pick`, `place`, `long-horizon`.
-   `--robot`: robot type used in simulation _(default: `franka`, also supports `piper`)_.
-   `--headless`: run simulation without GUI.
-   `--enable_cameras`: include visual observations in the output dataset.
-   `--debug`: enable debug mode and render trajectories as `.mp4` videos.
-   `--seed`: random seed for simulation _(automatically increments for each run if specified)_.
-   `-n`, `--n_simulations`: number of simulation episodes to generate *(default: `10,000`)*.
-   `--save`: save generated simulation data in HDF5 format.

### Trajectory Replay

After data generation, convert the trajectories into a format compatible with VLA training:

```bash
python3 scripts/translate_dataset_seq.py \
        --dataset_dir ../datasets --output_dir ../datasets-tr \
        --enable_cameras --headless --save
```

**Arguments:**

-   `--dataset_dir`: directory containing the raw simulation datasets.
-   `--output_dir`: directory to store the processed trajectories.
-   `--enable_cameras`: include visual observations in the output dataset.
-   `--headless`: run simulation without GUI.
-   `--save`: save generated simulation data in HDF5 format.

### Convert LeRobot Dataset

We provide a script to convert the generated `.h5` files into the LeRobot dataset *(v2.1 format)*, using **Euler angles** as the rotation representation:

```bash
python3 scripts/create_lerobot_dataset.py \  
        --dataset_dir ../datasets-tr --repo hzxie/DOM --rotation euler
```

This will create lerobot dataset using all the hdf5 datasets in the default output directory.

## Training 👩🏽‍💻

> ⚠️ This step requires the **PyTorch environment**

From the `PROJECT_ROOT/dynamic-vla` directory, run:

```bash
torchrun --nnodes=1  --nproc_per_node=8  --standalone run.py \  
  -c configs/dynamicvla.yaml \  
  -p /path/to/pretrained/model
  -d hzxie/DOM
```

**Arguments:**

-   `--nnodes`: number of compute nodes (machines) used for distributed training
-   `--nproc_per_node`: number of GPUs per node
-   `-c`: path to the training config file
-   `-p`: path to the pretrained model checkpoint *(optional)*
-   `-d`: name of the LeRobot dataset *(v2.1 format)*

### Checkpoint Evaluation

> ⚠️ This step requires the **PyTorch environment**

During training, you can automatically evaluate checkpoints using the following script:

```bash
python3 scripts/eval_checkpoints.py \
        -r euler -d -s -p "*fvit*46k*" \
        --ckpt_dir ./runs/checkpoints/ \  
```

**Arguments:**

-   `--ckpt_dir`: directory containing the checkpoints to be evaluated.
-   `-p`: pattern used to match checkpoint filenames *(supports wildcard patterns)*.
-   `--host`: host address of the evaluation server *(default: `127.0.0.1`)*.
-   `--img_port`: port used for the image stream on the evaluation server *(default: `3186`)*.
-   `--act_port`: port used for the action stream on the evaluation server *(default: `3188`)*.

## License 🗒️

This project is licensed under  [NTU S-Lab License 1.0](https://github.com/hzxie/DynamicVLA/blob/master/LICENSE). Redistribution and use should follow this license.
