#!/bin/bash

factor="optimisation"
model=$1
strategy=$2
dataset=$3
peft=$4

regenerate=0

runs=20
mruns=1

if [ $peft == 'lora' ]
then
    noise_layer='trainable'
    if [ $model == 'roberta' ]
    then
        lr=0.00001
    else
        lr=0.0001
    fi

elif [ $peft == 'ia3' ]
then
    noise_layer='trainable'
    lr=0.0075
elif [ $peft == 'unipelt' ]
then
    noise_layer='trainable'
    if [ $model == 'roberta' ]
    then
        lr=0.00025
    elif [ $model == 'albert' ]
    then
        lr=0.00006
    else
        lr=0.0001
    fi
else
    noise_layer='head'
    lr=0.00001
fi

echo $dataset

if [ $dataset == 'ag_news' ]
then
    max_len=50
elif [ $dataset == 'trec' ]
then
    max_len=25
elif [ $dataset == 'snips' ]
then
    max_len=25
elif [ $dataset == 'db_pedia' ] 
then
    max_len=50
elif [ $dataset == 'sst2' ] 
then
    max_len=50
elif [ $dataset == 'mrpc' ] 
then
    max_len=50
elif [ $dataset == 'boolq' ] 
then
    max_len=50
elif [ $dataset == 'cola' ] 
then
    max_len=25
fi
     
batch=8
experiment_name=fighting_randomness
    
echo $strategy

if [ $strategy == 'default' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=default --num_epochs=10 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --regenerate=$regenerate --peft=$peft --lr=$lr
if [ $strategy == 'all_data' ]
then
    if [ $dataset == 'db_pedia' ] || [ $dataset == 'ag_news' ]
    then
        labelled=55000
    else
        labelled=-1
    fi
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=default --num_epochs=10 --model=$model --batch_size=$batch --num_labelled=$labelled --max_len=$max_len --regenerate=$regenerate --peft=$peft --lr=$lr
elif [ $strategy == 'best_practices' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=longer --num_epochs=20 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --optimizer=AdamW --scheduler=linear_warmup --regenerate=$regenerate --peft=$peft --lr=$lr
elif [ $strategy == 'ensemble' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=ensemble --num_epochs=10 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --optimisation_mitigation=ensemble --ensemble_size=10 --regenerate=$regenerate --peft=$peft --lr=$lr
elif [ $strategy == 'input_noise' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=input_noise --num_epochs=10 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --optimisation_mitigation=noise --regenerate=$regenerate --peft=$peft --lr=$lr --noise_after_steps=25 --noise_location=input --noise_variance=0.15 --noise_type=scaled --noise_layer=$noise_layer --start_noise=0.3 --end_noise=0.9
elif [ $strategy == 'weights_noise' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=weights_noise --num_epochs=10 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --optimisation_mitigation=noise --regenerate=$regenerate --peft=$peft --lr=$lr --noise_after_steps=25 --noise_location=weights --noise_variance=0.15 --noise_type=scaled --noise_layer=$noise_layer --start_noise=0.3 --end_noise=0.9
elif [ $strategy == 'swa' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=swa --num_epochs=10 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --optimisation_mitigation=swa --optimizer=Adam --regenerate=$regenerate --peft=$peft --lr=$lr
elif [ $strategy == 'delayed_ensemble' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=de--num_epochs=10 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --optimizer=Adam --optimisation_mitigation=de --regenerate=$regenerate --peft=$peft --lr=$lr --noise_variance=0.15 --noise_type=scaled --noise_layer=$noise_layer --start_ensemble=0.9

elif [ $strategy == 'noisy_interpolation' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=ni--num_epochs=10 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --optimizer=Adam --optimisation_mitigation=ni --regenerate=$regenerate --peft=$peft --lr=$lr --noise_variance=0.15 --noise_type=scaled --noise_layer=$noise_layer --step_type=step --noise_after_steps=125 --ensemble_training_steps=125 --start_noise=0.3 --end_noise=0.6

elif [ $strategy == 'delayed_ensemble_with_noisy_interpolation' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=deni --num_epochs=10 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --optimizer=Adam --optimisation_mitigation=deni --regenerate=$regenerate --peft=$peft --lr=$lr --noise_variance=0.15 --noise_type=scaled --noise_layer=$noise_layer --step_type=step --noise_after_steps=125 --ensemble_training_steps=125 --start_noise=0.3 --end_noise=0.6 --start_ensemble=0.9
elif [ $strategy == 'delayed_ensemble_with_noisy_interpolation_and_augmented_labelled_samples' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=denials --num_epochs=10 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --optimizer=Adam --optimisation_mitigation=deni --regenerate=$regenerate --peft=$peft --lr=$lr --noise_variance=0.15 --noise_type=scaled --noise_layer=$noise_layer --step_type=step --noise_after_steps=125 --ensemble_training_steps=125 --start_noise=0.3 --end_noise=0.6 --start_ensemble=0.9 --augmented_data_size=1
elif [ $strategy == 'mixout' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=$strategy --num_epochs=10 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --optimizer=Adam --optimisation_mitigation=$strategy --regenerate=$regenerate --peft=$peft --lr=$lr
if [ $strategy == 'augment_1' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=$strategy --num_epochs=12 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --regenerate=$regenerate --peft=$peft --lr=$lr --augmented_data_size=1
if [ $strategy == 'augment_2' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=$strategy --num_epochs=12 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --regenerate=$regenerate --peft=$peft --lr=$lr --augmented_data_size=2
if [ $strategy == 'augment_3' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=$strategy --num_epochs=12 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --regenerate=$regenerate --peft=$peft --lr=$lr --augmented_data_size=3
if [ $strategy == 'augment_5' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=$strategy --num_epochs=12 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --regenerate=$regenerate --peft=$peft --lr=$lr --augmented_data_size=5
if [ $strategy == 'augment_10' ]
then
    python main.py --factor=$factor --mitigation_runs=$mruns --investigation_runs=$runs --dataset=$dataset --experiment_type=finetuning --experiment_name=$experiment_name --configuration_name=$strategy --num_epochs=12 --model=$model --batch_size=$batch --num_labelled=1000 --max_len=$max_len --regenerate=$regenerate --peft=$peft --lr=$lr --augmented_data_size=10
fi
    
