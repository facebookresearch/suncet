# PAWS  :paw_prints: **P**redicting View-**A**ssignments **w**ith **S**upport Samples

This repo provides a PyTorch implementation of:
* PAWS (**p**redicting view **a**ssignments **w**ith **s**upport samples), as described in the paper [Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples](https://arxiv.org/abs/2104.13963) (ICCV'21).
* RoPAWS (robust PAWS), as described in the paper [RoPAWS: Robust Semi-supervised Representation Learning from Uncurated Data](https://openreview.net/forum?id=G1H4NSATlr) (ICLR'23).

![CD21_260_SWAV2_PAWS_Flowchart_FINAL](https://user-images.githubusercontent.com/7530871/116110279-c82ff200-a672-11eb-9037-5c88d787f52e.png)

PAWS is a method for semi-supervised learning that builds on the principles of self-supervised distance-metric learning. PAWS pre-trains a model to minimize a consistency loss, which ensures that different views of the same unlabeled image are assigned similar pseudo-labels. The pseudo-labels are generated non-parametrically, by comparing the representations of the image views to those of a set of randomly sampled labeled images. The distance between the view representations and labeled representations is used to provide a weighting over class labels, which we interpret as a soft pseudo-label. By non-parametrically incorporating labeled samples in this way, PAWS extends the distance-metric loss used in self-supervised methods such as BYOL and SwAV to the semi-supervised setting.

RoPAWS extends PAWS to be robust when unlabeled data is uncurated, e.g., contains out-of-class data. To that end, RoPAWS reinterprets PAWS as a generative classifier that models densities on the representation space using kernel density estimation (KDE). From this probabilistic perspective, RoPAWS calibrates its prediction based on the densities of labeled and unlabeled data, which leads to a simple closed-form solution from the Bayes' rule. This simple modification makes PAWS robust under realistic uncuraed semi-supervised learning benchmarks.

Also provided in this repo is a PyTorch implementation of the semi-supervised SimCLR+CT method described in the paper [Supervision Accelerates Pretraining in Contrastive Semi-Supervised Learning of Visual Representations](https://arxiv.org/abs/2006.10803). SimCLR+CT combines the SimCLR self-supervised loss with the SuNCEt (supervised noise contrastive estimation) loss for semi-supervised learning.

## Pretrained models
We provide the full checkpoints for the PAWS pre-trained models, both with and without fine-tuning. The full checkpoints for the pretrained models contain the backbone, projection head, and prediction head weights. The finetuned model checkpoints, on the other hand, only include the backbone and linear classifier head weights.
Top-1 classification accuracy for the pretrained models is reported using a nearest neighbour classifier. Top-1 classification accuracy for the finetuned models is reported using the class labels predicted by the network's last linear layer.

<table>
  <tr>
    <th colspan="2"></th>
    <th colspan="2">1% labels</th>
    <th colspan="2">10% labels</th>
  </tr>
  <tr>
    <th>epochs</th>
    <th>network</th>
    <th>pretrained (NN)</th>
    <th>finetuned</th>
    <th>pretrained (NN)</th>
    <th>finetuned</th>
  </tr>
  <tr>
    <td>300</td>
    <td>RN50</td>
    <td><a href="https://dl.fbaipublicfiles.com/paws/paws_imgnt_1percent_300ep.pth.tar">64.2%</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/paws/paws_imgnt_1percent_300ep_finetuned.pth.tar">66.5%</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/paws/paws_imgnt_10percent_300ep.pth.tar">73.1%</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/paws/paws_imgnt_10percent_300ep_finetuned.pth.tar">75.5%</a></td>
  </tr>
  <tr>
    <td>200</td>
    <td>RN50</td>
    <td><a href="https://dl.fbaipublicfiles.com/paws/paws_imgnt_1percent_200ep.pth.tar">63.2%</a></td>
    <td>66.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/paws/paws_imgnt_10percent_200ep.pth.tar">71.9%</a></td>
    <td>75.0%</td>
  </tr>
  <tr>
    <td>100</td>
    <td>RN50</td>
    <td><a href="https://dl.fbaipublicfiles.com/paws/paws_imgnt_1percent_100ep.pth.tar">61.5%</a></td>
    <td>63.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/paws/paws_imgnt_10percent_100ep.pth.tar">71.0%</a></td>
    <td>73.9%</td>
  </tr>
</table>

## Running PAWS semi-supervised pre-training and fine-tuning

### Running PAWS vs. RoPAWS

To run RoPAWS, set `use_ropaws=True` and hyperparameters as in `configs/ropaws`. It changes the loss function `src/losses.py` to use RoPAWS version during training, and inference is the same as PAWS.

### Config files
All experiment parameters are specified in config files (as opposed to command-line-arguments). Config files make it easier to keep track of different experiments, as well as launch batches of jobs at a time. See the [configs/](configs/) directory for example config files.

### Requirements
* Python 3.8
* PyTorch install 1.7.1
* torchvision
* CUDA 11.0
* Apex with CUDA extension
* Other dependencies: PyYaml, numpy, opencv, submitit

### Labeled Training Splits
For reproducibilty, we have pre-specified the labeled training images as `.txt` files in the [imagenet_subsets/](imagenet_subsets/) and [cifar10_subsets/](cifar10_subsets/) directories.
Based on your specifications in your experiment's config file, our implementation will automatically use the images specified in one of these `.txt` files as the set of labeled images. On ImageNet, if you happen to request a split of the data that is not contained in [imagenet_subsets/](imagenet_subsets/) (for example, if  you set `unlabeled_frac !=0.9 and unlabeled_frac != 0.99`, i.e., not 10% labeled or 1% labeled settings), then the code will independently flip a coin at the start of training for each training image with probability `1-unlabeled_frac` to determine whether or not to keep the image's label.

### Single-GPU training
PAWS is very simple to implement and experiment with. Our implementation starts from the [main.py](main.py), which parses the experiment config file and runs the desired script (e.g., paws pre-training or fine-tuning) locally on a single GPU.

#### CIFAR10 pre-training
For example, to pre-train with PAWS on CIFAR10 locally, using a single GPU using the pre-training experiment configs specificed inside [configs/paws/cifar10_train.yaml](configs/paws/cifar10_train.yaml), run:
```
python main.py
  --sel paws_train
  --fname configs/paws/cifar10_train.yaml
```

#### CIFAR10 evaluation
To fine-tune the pre-trained model for a few optimization steps with the SuNCEt (supervised noise contrastive estimation) loss on a single GPU using the pre-training experiment configs specificed inside [configs/paws/cifar10_snn.yaml](configs/paws/cifar10_snn.yaml), run:
```
python main.py
  --sel snn_fine_tune
  --fname configs/paws/cifar10_snn.yaml
```
To then evaluate the nearest-neighbours performance of the model, locally, on a single GPU, run:
```
python snn_eval.py
  --model-name wide_resnet28w2
  --pretrained $path_to_pretrained_model
  --unlabeled_frac $1.-fraction_of_labeled_train_data_to_support_nearest_neighbour_classification
  --root-path $path_to_root_datasets_directory
  --image-folder $image_directory_inside_root_path
  --dataset-name cifar10_fine_tune
  --split-seed $which_prespecified_seed_to_split_labeled_data
```

#### CIFAR10 data setup
When setting up your CIFAR10 data, note the following relevant items in the config:
- `root_path` is the datasets directory where you put all your data
- `image_folder` is the folder inside `root_path` where `cifar-10-batches-py` exists (`cifar-10-batches-py` is the folder `torchvision.CIFAR10` looks for)

Here is __an example__ to setup your CIFAR10 data:

First [download](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) CIFAR10 and unzip; you will get a `cifar-10-batches-py` directory.
Then create the following directory structure:
```
|- datasets/
|-- cifar10-data/
|---- cifar-10-batches-py/
```
Finally, in your config specify:
- `root_path: datasets/`
- `image_folder: cifar10-data/`

You should now be able to run CIFAR10 experiments.

### Multi-GPU training
Running PAWS across multiple GPUs on a cluster is also very simple. In the multi-GPU setting, the implementation starts from [main_distributed.py](main_distributed.py), which, in addition to parsing the config file and launching the desired script, also allows for specifying details about distributed training. For distributed training, we use the popular open-source [submitit](https://github.com/facebookincubator/submitit) tool and provide examples for a SLURM cluster, but feel free to edit [main_distributed.py](main_distributed.py) for your purposes to specify a different approach to launching a multi-GPU job on a cluster.

#### ImageNet pre-training
For example, to pre-train with PAWS on 64 GPUs using the pre-training experiment configs specificed inside [configs/paws/imgnt_train.yaml](configs/paws/imgnt_train.yaml), run:
```
python main_distributed.py
  --sel paws_train
  --fname configs/paws/imgnt_train.yaml
  --partition $slurm_partition
  --nodes 8 --tasks-per-node 8
  --time 1000
  --device volta16gb
```

#### ImageNet fine-tuning
To fine-tune a pre-trained model on 4 GPUs using the fine-tuning experiment configs specified inside [configs/paws/fine_tune.yaml](configs/paws/fine_tune.yaml), run:
```
python main_distributed.py
  --sel fine_tune
  --fname configs/paws/fine_tune.yaml
  --partition $slurm_partition
  --nodes 1 --tasks-per-node 4
  --time 1000
  --device volta16gb
```
To evaluate the fine-tuned model locally on a single GPU, use the same config file, [configs/paws/fine_tune.yaml](configs/paws/fine_tune.yaml), but change `training: true` to `training: false`. Then run:
```
python main.py
  --sel fine_tune
  --fname configs/paws/fine_tune.yaml
```

### Soft Nearest Neighbours evaluation
To evaluate the nearest-neighbours performance of a pre-trained ResNet50 model on a single GPU, run:
```
python snn_eval.py
  --model-name resnet50 --use-pred
  --pretrained $path_to_pretrained_model
  --unlabeled_frac $1.-fraction_of_labeled_train_data_to_support_nearest_neighbour_classification
  --root-path $path_to_root_datasets_directory
  --image-folder $image_directory_inside_root_path
  --dataset-name $one_of:[imagenet_fine_tune, cifar10_fine_tune]
```

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation :paw_prints:
```
@article{assran2021semisupervised,
  title={Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples}, 
  author={Assran, Mahmoud, and Caron, Mathilde, and Misra, Ishan, and Bojanowski, Piotr and Joulin, Armand, and Ballas, Nicolas, and Rabbat, Michael},
  journal={arXiv preprint arXiv:2104.13963},
  year={2021}
}
```
```
@article{mo2023ropaws,
  title={RoPAWS: Robust Semi-supervised Representation Learning from Uncurated Data},
  author={Mo, Sangwoo and Su, Jong-Chyi and Ma, Chih-Yao and Assran, Mido and Misra, Ishan and Yu, Licheng and Bell, Sean},
  journal={arXiv preprint arXiv:2302.14483},
  year={2023}
}
```
```
@article{assran2020supervision,
  title={Supervision Accelerates Pretraining in Contrastive Semi-Supervised Learning of Visual Representations},
  author={Assran, Mahmoud, and Ballas, Nicolas, and Castrejon, Lluis, and Rabbat, Michael},
  journal={arXiv preprint arXiv:2006.10803},
  year={2020}
}
```
