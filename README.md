# EDB-Net
This project is the code implementation of the paper "EDB-Net: Entropy Dual-Branch Network for Improved Few-Shot Text Classification".
# Data
The datasets required for the experiment are shown in the table below. 
| Dataset | Avg. Length | Samples | Train / Valid / Test |
| ---- | ---- | ---- | ---- |
| HuffPost | 11.48 | 36900 | 20 / 5 / 16 |
| Amazon | 143.46 | 24000 | 10 / 5 / 9 |
| Reuters | 181.41 | 620 | 15 / 5 / 11 |
| 20News | 279.32 | 18828 | 8 / 5 / 7 |
| Banking77 | 11.77 | 13083 | 25 / 25 / 27 |
| HWU64 | 6.57 | 11036 | 23 / 16 / 25 |
| Liu57 | 6.66 | 25478 | 18 / 18 / 18 |
| Clinc150 | 8.31 | 22500 | 50 / 50 / 50 |
# Quick start
```key
conda create -n EDB python=3.7
source activate EDB
pip install -r requirements.txt
sh run.sh
``` 
Noting: before you start, you should download bert-base-uncased from https://huggingface.co/google-bert/bert-base-uncased, and change the path to your own file path.

