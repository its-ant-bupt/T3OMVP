# T3OMVP
The code of Transformer-based Team and Time Reinforcement Learning Scheme for Observation-constrained Multi-Vehicle Pursuit

## Prerequisites
- Linux or macOS or Windows
- Python 3.6
- CPU or NVIDIA GPU + CUDA CuDNN
### python modules
- gym==0.21.0
- numpy==1.19.5
- pytorch==1.8.0
- seaborn==0.9.0
- tensorboard==2.7.0
## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/pipihaiziguai/T3OMVP.git
cd T3OMVP
```
### Test the pre-trained model
```
python training_run.py --test --reload --reload_exp <name> --alg_name QMIX --UPDeT --lof 2 --history_length 5
```

### Train use T3OMVP
```
python training_run.py --alg_name QMIX --UPDeT --lof 2 --history_length 5
```
