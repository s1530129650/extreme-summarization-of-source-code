# A Convolutional Attention Network for Extreme Summarization of Source Code
## Env config

    conda create -n conv_cs python=3.6 -y
    conda activate conv_cs
    conda install pytorch=0.4.1 torchvision=0.2.1 cudatoolkit=11.0 -c pytorch
    pip install torchtext==0.2.3
    conda install numpy
    pip install six
 
or

```
yes | bash install_env.sh
```

## Data preprocessing

## Training

### Training inner project

    CUDA_VISIBLE_DEVICES=$4 python run_conv.py --project cassandra --seed 0 2>&1 |tee log/cassandra_res.txt

### Training inner project


### csn

    CUDA_VISIBLE_DEVICES=6 python run_conv.py --project csn --seed 0  --batch_size 64   --min_freq 100  2>&1 |tee log/csn_res.txt


CUDA_VISIBLE_DEVICES=6 python run_conv.py --project csn --seed 0 2>&1 |tee log/csn_res.txt

# ------------------- old readme-------------------
    
Implementation of [A Convolutional Attention Network for Extreme Summarization of Source Code](https://arxiv.org/abs/1602.03001) in PyTorch using TorchText

Using Python 3.6, PyTorch 0.4 and TorchText 0.2.3.

**Note**: only the *Convolutional Attention Model* currently works, the *Copy Convolutional Attention Model* is in progress.

To use:

1. `download.sh` to grab the dataset
1. `python preprocess.py` to preprocess the dataset into json format
1. `python run_conv.py` to run the Convolutional Attention Model with default parameters

Use `python run_conv.py -h` to see all the parameters that can be changed, e.g. to run the model on a different Java project within the dataset, use: `python run_conv.py --project {project name}`.


    ptional arguments:
      -h, --help            show this help message and exit
      --project PROJECT     Which project to run on (default: cassandra)
      --data_dir DATA_DIR   Where to find the training data (default: data)
      --checkpoints_dir CHECKPOINTS_DIR
                            Where to save the model checkpoints (default:
                            checkpoints)
      --no_cuda             Use this flag to stop using the GPU (default: False)
      --min_freq MIN_FREQ   Minimum times a token must appear in the dataset to
                            not be unk'd (default: 2)
      --batch_size BATCH_SIZE
      --emb_dim EMB_DIM
      --k1 K1
      --k2 K2
      --w1 W1
      --w2 W2
      --w3 W3
      --dropout DROPOUT
      --clip CLIP
      --epochs EPOCHS
      --seed SEED
      --load                Use this to load model parameters, parameters should
                            be saved as: {checkpoints_dir}/{project name}-conv-
                            model.pt (default: False)
