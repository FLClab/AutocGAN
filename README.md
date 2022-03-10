# AutocGAN

## Seting up the environment
1) Build the docker image
```docker build -t autogan .```

2) Run the docker image
```nvidia-docker run -it --rm --user $(id -u) --shm-size=10g --name=pytorch -v /:/workspace/ autogan```

Note : If the docker has access to multiple GPUs, use the following line in the command line to use only one GPU before running the experiment to avoid errors:

```export CUDA_VISIBLE_DEVICES=0```

## Searching for a cGAN architecture
1) Change the `topp`, `topk` and `num_candidate` (beam size) parameters in `autocgan.search.sh` for a search with the desired configuration. 
2) Top-K sampling is used by default. To use top-p, you need to set `topk 0`.
3) To launch the search:
``` sh exps/autocgan_search.sh ```

## Training the found cGAN architecture
Change the parameter ```--arch``` for the Top-1 architecture found from the searching phase in the file exps/derive_cgan.sh.
``` sh exps/derive_cgan.sh ```

## Testing the trained cGAN architecture
Change the parameter ```--arch``` and ```--load_path``` in the file ```exps/test_cgan.sh```
