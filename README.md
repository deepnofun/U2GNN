<p align="center">
	<img src="https://github.com/daiquocnguyen/U2GNN/blob/master/u2gnn_logo.png">
</p>

## Transformer for Graph Classification<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FU2GNN%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/U2GNN"><a href="https://github.com/daiquocnguyen/U2GNN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/U2GNN"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/U2GNN">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/U2GNN">
<a href="https://github.com/daiquocnguyen/U2GNN/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/U2GNN"></a>
<a href="https://github.com/daiquocnguyen/U2GNN/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/U2GNN"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/U2GNN">

This program provides the implementation of our U2GNN as described in our paper: [Universal Self-Attention Network for Graph Classification](https://arxiv.org/pdf/1909.11855.pdf) where we leverage on the transformer self-attention network. To the best of our knowledge, our work is the first consideration of using the unsupervised training setting to train a GNN-based model for the graph classification task. We show that a unsupervised model can noticeably outperform up-to-date supervised models by a large margin. Therefore, we suggest that future GNN works should pay more attention to the unsupervised training setting. This is important in both industry and academic applications in reality where expanding unsupervised models is more suitable due to the limited availability of class labels.

<p align="center">
	<img src="https://github.com/daiquocnguyen/U2GNN/blob/master/U2GNN.png">
</p>

## Usage

### Requirements
- Python 	3.x
- Tensorflow 	1.14
- Tensor2tensor 1.13
- Networkx 	2.3
- Scikit-learn	0.21.2

### Training

Regarding our unsupervised U2GNN:

	U2GNN$ python train_U2GNN_Unsup.py --dataset COLLAB --batch_size 512 --ff_hidden_size 1024 --num_neighbors 8 --num_sampled 512 --num_epochs 50 --num_hidden_layers 4 --learning_rate 0.00005 --model_name COLLAB_bs512_dro05_1024_8_idx0_4_3
	
	U2GNN$ python train_U2GNN_Unsup.py --dataset DD --batch_size 512 --ff_hidden_size 1024 --num_neighbors 4 --num_sampled 512 --num_epochs 50 --num_hidden_layers 3 --learning_rate 0.00005 --model_name DD_bs512_dro05_1024_4_idx0_3_3

Regarding our supervised U2GNN:

	U2GNN$ python train_U2GNN_Sup.py --dataset IMDBBINARY --batch_size 4 --ff_hidden_size 1024 --fold_idx 1 --num_neighbors 8 --num_sampled 512 --num_epochs 50 --num_hidden_layers 4 --learning_rate 0.0005 --model_name IMDBBINARY_bs4_fold1_dro05_1024_8_idx0_4_1
	
	U2GNN$ python train_U2GNN_Sup.py --dataset PTC --batch_size 4 --ff_hidden_size 1024 --fold_idx 1 --num_neighbors 16 --num_sampled 512 --num_epochs 50 --num_hidden_layers 3 --learning_rate 0.0005 --model_name PTC_bs4_fold1_dro05_1024_16_idx0_3_1

**Parameters:** 

`--learning_rate`: The initial learning rate for the Adam optimizer.

`--batch_size`: The batch size.

`--dataset`: Name of dataset.

`--num_epochs`: The number of training epochs.

`--num_hidden_layers`: The number T of timesteps.

`--fold_idx`: The index of fold in 10-fold validation.

**Notes:**

I fixed the number of stacked layers to 1 (i.e., k_num_GNN_layers=1) to show in our paper that our U2GNN aggregation function is a more advanced computation process. You should tune this hyper-parameter to have better results.

You can see command examples in `command_examples.txt`. You should only use `train_U2GNN_Unsup_large_dataset.py` and `eval_large_dataset.py` for a large collection of graphs such as REDDIT if having OOM or problems with Tensorflow when running `train_U2GNN_Unsup.py`.

## Cite  
Please cite the paper whenever U2GNN is used to produce published results or incorporated into other software:

	 @article{Nguyen2019U2GNN,
		  author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dinh Phung},
		  title={{Universal Self-Attention Network for Graph Classification}},
		  journal={arXiv preprint arXiv:1909.11855},
		  year={2019}
		  }

## License
As a free open-source implementation, U2GNN is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

U2GNN is licensed under the Apache License 2.0.
