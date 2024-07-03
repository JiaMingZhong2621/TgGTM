# TgGTM :Text-guided Graph Temporal Modeling for Few-Shot Video Classification
![GitHub Logo](/overview.png)
This code is based on [CLIP-FSAR](https://github.com/alibaba-mmai-research/CLIP-FSAR) and [trx](https://github.com/tobyperrett/trx)
.
# Installation
Requirements:
* Python == 3.11.4
* torch == 2.0.1
* torchvision == 0.15.2
* simplejson == 3.19.2
*	pyyaml
*	einops
*	oss2
*	psutil
*	tqdm
*	pandas
# Data preparation
First, download the datasets from their original source. (If you have already downloaded them, you can skip this step.)
* [SSV2](https://20bn.com/datasets/something-something)
* [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)
* [Kinetics](https://github.com/Showmax/kinetics-downloader)
* [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
Then, prepare data according to the [splits](https://github.com/JiaMingZhong2621/TgGTM/tree/main/splits) we provide. You can also refer to [CLIP-FSAR](https://github.com/alibaba-mmai-research/CLIP-FSAR?tab=readme-ov-file).
# Running
Before running, some settings need to be configured in the config file.
1. open utils/config.py
2. Set `<cfg.default="/media/psdz/新加卷/zjm/TgGTM/configs/projects/CLIPFSAR/ssv2_small/CLIPFSAR_SSv2_Small_1shot_v1.yaml">` to specify the configuration files used by the datasets.
3. you can then do:
   `python run_fsar.py`
   
Please ensure that the datasets used in `utils/config.py` are consistent with those used in `run_fsar.py`.
If you need to modify the learning rate, please do so in `run_fsar.py`. To change the number of iterations, modify the configurations in [configs](https://github.com/JiaMingZhong2621/TgGTM/tree/main/configs/projects/CLIPFSAR).
