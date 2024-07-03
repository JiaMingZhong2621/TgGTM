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
# Running
Before running, some settings need to be configured in the config file.
1.configs/projects/TgGTM/kinetics100/CLIPFSAR_K100_1shot_v1.yaml
