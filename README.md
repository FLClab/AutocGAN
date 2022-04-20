# AutocGAN

## Seting up the environment
1) Build the docker image
```docker build -t autogan .```

2) Run the docker image
```nvidia-docker run -it --rm --user $(id -u) --shm-size=10g --name=pytorch -v /:/workspace/ autogan```

Note : If the docker has access to multiple GPUs, use the following line in the command line to use only one GPU before running the experiment to avoid errors:

```export CUDA_VISIBLE_DEVICES=0```

We have been using CUDA V10.0.130 and V10.1.243 for this work.

## Searching for a cGAN architecture
1) Change the `topp`, `topk` and `num_candidate` (beam size) parameters in `exps/autocgan_search.sh` for a search with the desired configuration. 
2) Top-K sampling is used by default. To use top-p, you need to set `topk 0`.
3) To launch the search:
``` sh exps/autocgan_search.sh ```

## Training the discovered cGAN architecture
### Download or compute the precalculated statistics for the FID score
The precalculated statistics for CIFAR-10 and SVHN are available [on the website of the Institute of Bioinformatics of the Johannes Kepler University](http://bioinf.jku.at/research/ttur/). The .npz file must be downloaded and included in the fid_stat folder prior to training, since it is required to compute the FID score. For a different dataset, the statistics can be computed using the cal_fid_stat.py function in utils. Specify the options ```--data_path``` (should point to a folder that contains all the images in .png format) and ```output_file``` (the title of the resulting .npz file).

### Train the discovered architecture
1) Change the parameter ```--arch``` for the Top-1 architecture found from the searching phase in the file exps/derive_cgan.sh.
2) Use a descriptive name for the `exp_name` parameter, as it will be used for testing.  
3) To launch training: 
``` sh exps/derive_cgan.sh ```

## Testing the trained cGAN architecture
1) Change the parameter ```--arch``` and ```--load_path``` in the file ```exps/test_cgan.sh```.   
2) The load path corresponds to the name given to the training experiment.   
For example, if you set `exp_name derive_cgan_demo` for the training, then the load path here should be `logs/derive_cgan_demo/Model/checkpoint_best.pth`  
3) To launch testing:    
```sh exps/test_cgan.sh```

## Pre-configured experiment files

We provide pre-configured experiment files to reproduce the work published in the paper. 
- exps/autocgan_search_cifar10_topk.sh : search for an architecture using top-K sampling for the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset
- exps/autocgan_search_cifar100_topp.sh : search for an architecture using top-P sampling for the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset
- exps/autocgan_search_stl_topp.sh : search for an architecture using top-P sampling for the [STL10](https://cs.stanford.edu/~acoates/stl10/) dataset, using the original size (96x96)
- exps/autocgan_search_stl_topp_resize48.sh : search for an architecture using top-P sampling for the STL10 dataset, resized to half the original size (48x48)
