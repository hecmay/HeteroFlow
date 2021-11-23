
# HeteroFlow [![DOI](https://zenodo.org/badge/327196438.svg)](https://zenodo.org/badge/latestdoi/327196438)

Code in this repo is for artifact evaluation of the paper HeteroFlow: An Accelerator Programming Model with Decoupled Data Placement for Software-Defined FPGAs

**For the compiler source code and pre-trained model release, please refer to our [repository](https://github.com/cornell-zhang/heterocl/tree/heteroflow)**. In the repo, we only include the dataset, compiler-optimized HLS code, and pre-compiled bitstream.


## Structure
```

```

## Dependency

```

```

## Usage
First, make sure to import the right model. This can be done by choosing the ```candidates``` in **line 27** in the training script ```cifar10.py```.

### Software Emulation:
- Run ``

### Hardware Execution:
- Step 1: setup the en

## Results
Dataset: CIFAR-10; Model: ResNet-20
| Model         | Top-1 %   |
| ------------- | --------- |
| binput; W/A=1/1.4 (PG)  | 89.1      |
