# DeepCNA

DeepCNA is a computational method for detecting single-cell copy number alterations from scDNA-seq data.

## Requirements

* Python 3.8+.

# Installation
## Clone repository
First, download DeepCNA from github and change to the directory:
```bash
git clone https://github.com/zhyu-lab/deepcna
cd deepcna
```

## Create conda environment (optional)
Create a new environment named "deepcna":
```bash
conda create --name deepcna python=3.8.15
```

Then activate it:
```bash
conda activate deepcna
```

## Install requirements
To successfully compile the source code, please make sure g++-5 is installed on your machine.
you can install it using following commands:
```bash
sudo apt install gcc-5 g++-5
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 20  
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 20
```
If your current compiler is not g++-5, you need to change the current compiler to gcc-5 and g++-5:
```bash
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```
Then install related packages and compile the code:
```bash
python -m pip install -r ./detectbp/requirements.txt
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
cd prep
cmake .
make
cd ..
chmod +x run_deepcna.sh ./callcna/run_callcna.sh ./callcna/callcna
```
After the installation completes, you can reset the compiler using same commands:
```bash
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```
# Usage

## Step 1: prepare input data

We use same pipeline as used in [rcCAE](https://github.com/zhyu-lab/rccae) to prepare the input data, please refer to [rcCAE](https://github.com/zhyu-lab/rccae) for detailed instructions. 

## Step 2: train the autoencoder model

The “./detectbp/train.py” Python script is used to learn latent representations of bins and infer breakpoints.

The arguments to run “./detectbp/train.py” are as follows:

Parameter | Description | Possible values
---- | ----- | ------
--input | input file generated by “./prep/bin/prepInput” command | Ex: /path/to/example.txt
--output | a directory to save results | Ex: /path/to/results
--epochs | number of epoches to train the AE | Ex: epochs=200  default:500
--batch_size | batch size | Ex: batch_size=64  default:256
--lr | learning rate | Ex: lr=0.0005  default:0.0001
--latent_dim | latent dimensionality | Ex: latent_dim=1  default:2
--min_size | minimum size of segments | Ex: min_size=5  default:3
--seed | random seed | Ex: seed=1  default:0

Example:

```
python ./detectbp/train.py --input ./data/example.txt --epochs 500 --batch_size 256 --lr 0.0001 --latent_dim 1 --seed 0 --output data
```

## Step 3: detect single-cell CNAs

The “./callcna/callcna.m” MATLAB script is used to call single-cell copy numbers. 

The arguments run “./callcna/callcna.m” are as follows:

Parameter | Description | Possible values
---- | ----- | ------
lrcFile | “lrc.txt” file generated by “./detectbp/train.py” script | Ex: /path/to/lrc.txt
segFile | seg.txt” file generated by “./detectbp/train.py” script | Ex: /path/to/seg.txt
outputDir | a directory to save results | Ex: /path/to/results
maxCN | maximum copy number to consider | Ex: 6  default:10

Example:

```
callcna('../data/lrc_example.txt','../data/seg_example.txt','../data',10)
```

**We also provide a script “run_deepcna.sh” to integrate all three steps to run DeepCNA.**
This script requires that MATLAB Compiler Runtime (MCR) v91 (R2016b) is installed in user's machine. 
The MCR can be downloaded from [MathWorks Web site](https://www.mathworks.com/products/compiler/matlab-runtime.html). 

Example:

```
./run_deepcna.sh /path/to/bam /path/to/ref.fa /path/to/ref.bw /path/to/barcodes.txt /path/to/results
```
Type ./run_deepcna.sh to learn details about how to use this script.

# Contact

If you have any questions, please contact lfr_nxu@163.com.