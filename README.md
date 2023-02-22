# Code of AccelGSCAD
This is the Cython code of Group SCAD/Fast Group SCAD/AccelGSCAD.

# Files
- `README.md`
- `setup.py`
- `skip_setup.py`
- `fast_setup.py`
- `group_scad.pyx`
- `skip_group_scad.pyx`
- `fast_group_scad.pyx`
- `sgl_tools.py`
- `experiment.py`
- `grid_experiments.sh`

# Environment
The environment can be made by using the Dockerfile of Kaggle as follows:

```
https://github.com/Kaggle/docker-python
```

You may need to write the following additional lines in the Dockerfile.

```
RUN pip install ipdb
RUN conda install -c anaconda cython
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install sklearn
```

We recommend you to use Python 3.8 and Cython 2.9.

# Datasets
- eunite2001
  - You can download the dataset from LIBSVM cite.
  - Please put the dataset under `./data`.

```
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/eunite2001
```

- triazines
  - You can download the dataset from LIBSVM cite.
  - Please put the dataset under `./data`.

```
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/triazines_scale
```

- The other datasets are automatically downloaded from OpenML cite.

# Compile
We can compile our cython codes on the above environment as follows:

- Group SCAD (Breheny et al., Statistics and Computing, 2015.)

```
python3 setup.py build_ext --inplace
```

- Fast Group SCAD (Ida et al., NeurIPS, 2019.)

```
python3 skip_setup.py build_ext --inplace
```

- AccelGSCAD (ours)

```
python3 fast_setup.py build_ext --inplace
```

# Usage

- Please `perform grid_experiments.sh` on the above docker environment.

```
bash grid_experiments.sh
```

- The above command generates files of the processing times, the objective values, the losses, the parameters and the logs.

- Each method on the datasets of `eunite` and `qsbralks` can be performed in less than a minute on one CPU core of 2.20 GHz Intel Xeon server running Linux.

- Each method on the datasets of `qsbr_rw1`, `qsf` and `triazines` is commented out in the code of `grid_experiments.sh` because it takes more than 10 minuetes. Please see the code if you want to perform the methods on these datasets.

# Note
- If you want to change the hyperparameter `m` in our method, you can rewrite `line 129` in the code of `fast_group_scad.pyx` and re-compile it.
- If you want to evaluate the exact processing time, please specify the core ID of the CPU (logical core, not physical core) on docker, e.g. `--cpuset-cpus=0,20`. In this case, please specify the logical core IDs so that they do not cross the physical cores.
