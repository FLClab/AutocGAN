# AutocGAN

## Seting up the environment
1) Build the docker image
```docker build -t autogan .```

2) Run the docker image
```nvidia-docker run -it --rm --user $(id -u) --shm-size=10g --name=pytorch -v /:/workspace/ autogan```

3) Run an experiment
```sh exps/autocgan_search.py```

Note : If the docker has access to multiple GPUs, use the following line in the command line to use only one GPU before running the experiment to avoid errors:

```export CUDA_VISIBLE_DEVICES=0```

## Searching for a cGAN architecture
``` sh exps/autocgan_search.sh ```

## Training the found cGAN architecture
Change the parameter ```--arch``` for the Top-1 architecture found from the searching phase in the file exps/derive_cgan.sh.
``` sh exps/derive_cgan.sh ```

## Testing the trained cGAN architecture
Change the parameter ```--arch``` and ```--load_path``` in the file ```exps/test_cgan.sh```
