# Fighting Randomness with Randomness: Mitigating Optimisation Instability of Fine-Tuning using Delayed Ensemble and Noisy Interpolation

This repository contains the experiments and the proposed mitigation method for the paper: "Fighting Randomness with Randomness: Mitigating Optimisation Instability of Fine-Tuning using Delayed Ensemble and Noisy Interpolation" accepted at EMNLP'24 findings ([preprint](https://arxiv.org/abs/2406.12471)).

## Dependencies and local setup

The code in this repository uses Python. The required dependencies are specified in the `requirements.txt`. 

Simply run `pip install -r requiremets.txt`.

## Running the mitigation strategy

To run the specific mitigation strategy, follow these steps:

1. Make sure all the requirements are installed.
1. Choose dataset to run the mitigation strategy on. Currently we support following options: "sst2", "mrpc", "cola", "boolq", "rte", "trec", "ag_news", "snips", "db_pedia". However, as we use the HuggingFace, the set of datasets can be easily extended to include other ones (the dataset classes in `data.py` file needs to be extended with the loading and processing of the new dataset). 
1. Choose the training dataset size to run the investigation on using the `labelled` parameter.
1. Choose number of runs for the experiments. The `investigation_runs` parameter specifies the number of random seeds that will be used for the mitigation (changing initialisation, order of samples and model randomness), while the `mitigation_runs` specifies the number of runs for the remaining randomness factors (data split for splitting data and label choice for selecting what samples are considered to be labelled)
1. Choose the mitigation strategy to run. Currently we support following options: "default", "ensemble", "noise", "prior_noise", "swa", "ema", "mixout", "delayed_ensemble" (or "de"), "noisy_interpolation" (or "ni"), "delayed_ensemble_with_noisy_interpolation" (or "deni").
1. Choose the model and peft method to run.
1. Choose the setup for mitigation strategy hyperparameters.
1. Run the investigation using following command (with SST-2 dataset, BERT with IA3 PEFT method, on a subset of 1 000 training samples, with 20 mitigation runs and the default baseline):
    ```bash
    python main.py --factor=optimisation --mitigation_runs=20 --investigation_runs=1 --dataset=sst2 --experiment_type=finetuning --experiment_name=mitigating_randomness --configuration_name=default --num_epochs=10 --model=bert --batch_size=8 --num_labelled=1000 --max_len=50  --peft=lora 
    ```
1. The results from these runs will be saved to the folder specified by the `experiment_name`, `configuration_name`, `experiment_type`, `model`, `dataset` and `factor` arguments. The above command will save the results into the following path: `results/mitigating_randomness/finetuning_lora_bert_base/default/sst2/optimisation`. After the experiments are run, this folder should contain 1 folder `mitigation_0` containing 20 folders `investigation_{idx}` with idx ranging 0-19 which contain the results.

To allow for reproducibility we provide a script to run all the experiments from our paper, with the hyperparameters set -- `run_randomness_fight.sh`. The script takes 4 arguments -- `model`, `mitigation_strategy`, `dataset` and `peft`.

## Evaluating the experiments

For the evaluation purposes, we provide python script `process_randomness_fight_results.py` that will process all the results and create a single csv file `full_results.csv`.


## Paper Citing

```
@inproceedings{pecher-etal-2024-fighting,
    title = "On Sensitivity of Learning with Limited Labelled Data to the Effects of Randomness: Impact of Interactions and Systematic Choices",
    author = "Pecher, Branislav  and
      Cegin, Jan and
      Belanec, Robert and
      Simko, Jakub and
      Srba, Ivan  and
      Bielikova, Maria",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    year = "2024",
    publisher = "Association for Computational Linguistics",
}
```