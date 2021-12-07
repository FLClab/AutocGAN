# AutocGAN

1) Build the docker image
```docker build -t autogan .```

2) Run the docker image
```nvidia-docker run -it --rm --user $(id -u) --shm-size=10g --name=pytorch -v /:/workspace/ autogan```

3) Run an experiment
```sh exps/autogan_search.py```

Note : There are weird errors when using multiple GPUs. Specify this in the command line before running the experiment to use only one GPU and avoid these errors.

```export CUDA_VISIBLE_DEVICES=1```
