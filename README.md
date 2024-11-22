# Continual Learning for LLMs

## Setting Up the Environment
(I am assuming that you have anaconda installed, if not please install it.)
Clone the repository and follow the steps below. Open the terminal and type:
- `conda create --name nlp(or you name it) --file environment.txt`
then type:
- `pip install -r environment.yml`

You should now have a conda environment with name "nlp" that contains the required python packages to run the repo.

## Explanation of the Python Scripts

The Python scripts are in the **codebase** folder. 

#### Main Python Scripts
Note, if the file ends with `_acc.py` it implies the script has DeepSpeed training through Accelerate.
- `main_joint_acc.py.py` carries out the joint training with the specified options. (Upper Bound Method)
- `main_olora_acc.py.py` does the O-LoRA training with the specified options.
- `main_olora_acc_v2.py.py` does the O-LoRA training but without merging the O-LoRA parameters in the continual training.
- `main_seq_acc.py` does sequential training with the specified options. (Lower Bound Method)
- `instruction_tuning_script_GPT2.py` is not used anymore please ignore it.

#### Other Python Scripts
- `utils.py` includes some utility functions such as dataset preprocessing functions, model and tokenizer returns etc.
- `dataloaders.py` has functions to load the datasets from CL_BENCHMARK. Currently it only loads dbpedia, amazon, yahoo, agnews. If you want to load another dataset, you need to write a new function for it. This script can be made more modular actually, so you can also refine the code if you want.
- `custom_data_collator.py` our custom collator function.
- `options.py` contains the arguments that can be specified for running any of the main Python scripts. Please take a look at what arguments are available in it.

## Deepspeed/Accelerate Configuration Files
- `default_config.yaml` is the config file for the *Accelerate* library. In the file, you should set the value of `num_processes` to the number of GPUs you have. (If you have more machines to run the experiment with, change `num_machines` accordingly as well.) This script also tells Accelerate to use DeepSpeed plugin, the specification of which are available in another file.
- `ds_config_stage2.json`. We are using DeepSpeed stage 2 which offloads model parameters to the CPU RAM. You do not really need to change anything here except for batch size keys. Please set `train_micro_batch_size_per_gpu` the batch size that you pass when running the bash script. Please also change `train_batch_size` accordingly, which should be the number of GPUs (processes) times `train_micro_batch_size_per_gpu`.

## Bash Files to run Experiments
- `sequential.sh`, `joint.sh`, `olora.sh` files contain example experiment settings that can be used to run sequential, joint and O-LoRA training respectively.
- `olora_hpc.sh` has a sample O-LoRA training script that can be run on HPC. Based on your allowed nodes, please modify the file first and then submit. Also do not forget to make the `ds_config_stage2.json` aligned with your batch size selection!