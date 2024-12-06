![INN](/Figure1.png)


## Interpolating Neural Network

This is the github repo for the paper ["Interpolating neural network (INN): A novel unification of machine learning and interpolation theory"](https://arxiv.org/abs/2404.10296).

INN is a lightweight yet precise network architecture that can replace MLPs for data training, partial differential equation (PDE) solving, and parameter calibration. The key features of INNs are:

* Less trainable parameters than MLP without sacrificing accuracy
* Faster and proven convergent behavior
* Fully differntiable and GPU-optimized


## Installation


Clone the repository:

```bash
git clone https://github.com/hachanook/pyinn.git
cd pyinn
```

Create a conda environment:

```bash
conda clean --all # [optional] to clear cache files in the base conda environment
conda env create -f environment.yaml
or
conda install -n base -c conda-forge mamba # [optional] install mamba in the base conda environment
mamba env create -f environment.yaml # this makes installation faster

conda activate pyinn-env
```

Install JAX
- See jax installation [instructions](https://github.com/jax-ml/jax?tab=readme-ov-file#installation). Depending on your hardware, you may install the CPU or GPU version of JAX. Both will work, while GPU version usually gives better performance.
- For CPU only (Linux/macOS/Windows), one can simply install JAX using:
```bash
pip install -U jax
```
- For GPU (NVIDIA, CUDA 12)
```bash
pip install -U "jax[cuda12]"
```
- For TPU (Google Cloud TPU VM)
```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Install Optax (optimization library of JAX)
```bash
pip install optax
```


Then there are two options to continue:

### Option 1

Install the package locally:

```bash

pip install -e .
```

### Option 2

Install the package from the [PyPI release](https://pypi.org/project/pyinn/0.1.0/) directly:

```bash
pip install pyinn
```

### Quick test

```bash
python ./pyinn/main.py
```

## License
This project is licensed under the GNU General Public License v3 - see the [LICENSE](https://www.gnu.org/licenses/) for details.

## Citations
If you found this library useful in academic or industry work, we appreciate your support if you consider 1) starring the project on Github, and 2) citing relevant papers:

```bibtex
@article{park2024engineering,
  title={Engineering software 2.0 by interpolating neural networks: unifying training, solving, and calibration},
  author={Park, Chanwook and Saha, Sourav and Guo, Jiachen and Zhang, Hantao and Xie, Xiaoyu and Bessa, Miguel A and Qian, Dong and Chen, Wei and Wagner, Gregory J and Cao, Jian and others},
  journal={arXiv preprint arXiv:2404.10296},
  year={2024}
}
```