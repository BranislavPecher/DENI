from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import transformers
from datasets import Dataset
from data import FineTuningDataset, DatasetLoader
from transfer_learning.models import BERTBase, RoBERTaBase, ALBERTBase, PEFTBERTBase, PEFTRoBERTaBase, PEFTALBERTBase
from mixout import MixLinear
import random
import pickle
import argparse
import torch
import numpy as np
import os
import copy
import json
import time
import torch.nn.functional as F
import pandas as pd
from deterministic_noise import set_rng_state, restore_rng_state, get_rng_state

def SWA_ft_experiment(dataset, trainloader, testloader, model_initialisation_seed, model_randomness_seed):
    net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed)
    net.cuda()
    if OPTIMISER == 'Adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    elif OPTIMISER == 'AdamW':
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE, weight_decay=0 if SCHEDULER != 'default' else 0.01)
    num_steps = NUM_EPOCHS * len(trainloader)
    if SCHEDULER != 'default':
        if 'warmup' in SCHEDULER:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
        else:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    net.train()
    swa_model = torch.optim.swa_utils.AveragedModel(net)
    swa_start = int(num_steps * 0.25)

    step = 0
    for epoch in range(NUM_EPOCHS):
        for batch_idx, data in enumerate(trainloader):
            step += 1
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = net(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()
            
            
            if step >= swa_start:
                swa_model.update_parameters(net)
            if SCHEDULER != 'default':
                scheduler.step()

    torch.optim.swa_utils.update_bn(trainloader, swa_model)
    
    golden = []
    predictions = []
    
    swa_model.eval()
    for batch_idx, data in enumerate(testloader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = swa_model(ids, mask, token_type_ids)

        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        golden.extend(data['targets'].tolist())
    return golden, predictions


def EMA_ft_experiment(dataset, trainloader, testloader, model_initialisation_seed, model_randomness_seed):
    net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed)
    net.cuda()
    if OPTIMISER == 'Adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    elif OPTIMISER == 'AdamW':
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE, weight_decay=0 if SCHEDULER != 'default' else 0.01)
    num_steps = NUM_EPOCHS * len(trainloader)
    if SCHEDULER != 'default':
        if 'warmup' in SCHEDULER:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
        else:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    net.train()
    ema_model = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    
    for epoch in range(NUM_EPOCHS):
        for batch_idx, data in enumerate(trainloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = net(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            ema_model.update_parameters(net)
            if SCHEDULER != 'default':
                scheduler.step()
            

    torch.optim.swa_utils.update_bn(trainloader, ema_model)
    
    golden = []
    predictions = []
    
    net.eval()
    for batch_idx, data in enumerate(testloader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = ema_model(ids, mask, token_type_ids)

        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        golden.extend(data['targets'].tolist())
    return golden, predictions


def default_experiment(dataset, trainloader, testloader, model_initialisation_seed, model_randomness_seed, return_preds=False):
    print('Creating model')
    net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed)
    net.cuda()
    if OPTIMISER == 'Adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    elif OPTIMISER == 'AdamW':
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE, weight_decay=0 if SCHEDULER != 'default' else 0.01)
    
    num_steps = NUM_EPOCHS * len(trainloader)
    if SCHEDULER != 'default':
        if 'warmup' in SCHEDULER:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
        else:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    net.train()
    print('Model and scheduler created')
    for epoch in range(NUM_EPOCHS):
        # print(f'Epoch: {epoch}')
        for batch_idx, data in enumerate(trainloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = net(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()
            if SCHEDULER != 'default':
                scheduler.step()
    
    golden = []
    predictions = []
    output_preds = []
    
    net.eval()
    for batch_idx, data in enumerate(testloader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = net(ids, mask, token_type_ids)
        if return_preds:
            output_preds.extend(outputs.data.tolist())

        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        golden.extend(data['targets'].tolist())
    return golden, predictions, output_preds


def mixout_experiment(dataset, trainloader, testloader, model_initialisation_seed, model_randomness_seed):
    net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed)
    for name, module in net.named_modules():
        if name in ['dropout'] and isinstance(module, torch.nn.Dropout):
            setattr(net, name, torch.nn.Dropout(0))
            print('Replacing Dropout layer!')
        if name in ['output'] and isinstance(module, torch.nn.Linear):
            target_state_dict = module.state_dict()
            bias = True if module.bias is not None else False
            new_module = MixLinear(module.in_features, module.out_features, bias, target_state_dict['weight'], 0.9)
            new_module.load_state_dict(target_state_dict)
            setattr(net, name, new_module)
            print('Replacing Linear layer with MixLinear!')
    net.cuda()
    if OPTIMISER == 'Adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    elif OPTIMISER == 'AdamW':
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE, weight_decay=0 if SCHEDULER != 'default' else 0.01)
    
    num_steps = NUM_EPOCHS * len(trainloader)
    if SCHEDULER != 'default':
        if 'warmup' in SCHEDULER:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
        else:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    net.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, data in enumerate(trainloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = net(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()
            if SCHEDULER != 'default':
                scheduler.step()
    
    golden = []
    predictions = []
    
    net.eval()
    for batch_idx, data in enumerate(testloader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = net(ids, mask, token_type_ids)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        golden.extend(data['targets'].tolist())
    return golden, predictions


def noise_regularisation_experiment(dataset, trainloader, testloader, model_initialisation_seed, model_randomness_seed, noise_params):
    net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed)
    net.cuda()
    if OPTIMISER == 'Adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    elif OPTIMISER == 'AdamW':
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE, weight_decay=0 if SCHEDULER != 'default' else 0.01)
    
    num_steps = NUM_EPOCHS * len(trainloader)
    if SCHEDULER != 'default':
        num_steps = NUM_EPOCHS * len(trainloader)
        if 'warmup' in SCHEDULER:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
        else:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_steps)
    loss_fn = torch.nn.CrossEntropyLoss()
    print(f'Regular noise experiment with type {noise_params["noise_type"]} in layer {noise_params["noise_layer"]}')
    net.train()

    step = 1
    steps_since_last = 0
    for epoch in range(NUM_EPOCHS):
        for batch_idx, data in enumerate(trainloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            optimizer.zero_grad()
            if step >= int(noise_params['start_noise'] * num_steps) and step < int(noise_params['end_noise'] * num_steps) and steps_since_last > noise_params['noise_after_steps']:
                states = get_rng_state()
                restore_rng_state(NOISE_PARAMS['states'])
                steps_since_last = 0
                if noise_params['noise_location'] == 'input':
                    outputs = net(ids, mask, token_type_ids, noise_params['noise_variance'])
                else:
                    with torch.no_grad():
                        if noise_params['noise_layer'] == 'head':
                            param = net.output.weight
                            param.data += ((noise_params['noise_variance'] / step)**0.5) * torch.randn(param.shape).cuda() * (torch.std(param.data) if noise_params['noise_type'] == 'scaled' else 1)
                            param = net.output.bias
                            param.data += ((noise_params['noise_variance'] / step)**0.5) * torch.randn(param.shape).cuda() * (torch.std(param.data) if noise_params['noise_type'] == 'scaled' else 1)
                        else:
                            for param in net.parameters():
                                if param.requires_grad == True:
                                    param.data += ((noise_params['noise_variance'] / step)**0.5) * torch.randn(param.shape).cuda() * (torch.std(param.data) if noise_params['noise_type'] == 'scaled' else 1)
                
                    outputs = net(ids, mask, token_type_ids)
                NOISE_PARAMS['states'] = get_rng_state()
                restore_rng_state(states)
            else:
                steps_since_last += 1
                outputs = net(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()
            if SCHEDULER != 'default':
                scheduler.step()
            step += 1
    
    golden = []
    predictions = []
    
    net.eval()
    for batch_idx, data in enumerate(testloader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = net(ids, mask, token_type_ids)

        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        golden.extend(data['targets'].tolist())
    return golden, predictions


def prior_noise_regularisation_experiment(dataset, trainloader, testloader, model_initialisation_seed, model_randomness_seed, noise_params):
    net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed)
    net.cuda()
    if OPTIMISER == 'Adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    elif OPTIMISER == 'AdamW':
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE, weight_decay=0 if SCHEDULER != 'default' else 0.01)
    
    num_steps = NUM_EPOCHS * len(trainloader)
    if SCHEDULER != 'default':
        if 'warmup' in SCHEDULER:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
        else:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_steps)
    loss_fn = torch.nn.CrossEntropyLoss()
    print(f'Prior noise experiment with type {noise_params["noise_type"]} in layer {noise_params["noise_layer"]}')
    with torch.no_grad():
        states = get_rng_state()
        restore_rng_state(NOISE_PARAMS['states'])
        for param in net.parameters():
            if param.requires_grad == True:
                param.data += ((noise_params['noise_variance'])**0.5) * torch.randn(param.shape).cuda() * (torch.std(param.data) if noise_params['noise_type'] == 'scaled' else 1)    
        NOISE_PARAMS['states'] = get_rng_state()
        restore_rng_state(states)
        
    net.train()
    step = 1
    for epoch in range(NUM_EPOCHS):
        for batch_idx, data in enumerate(trainloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            optimizer.zero_grad()
            
            outputs = net(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()
            if SCHEDULER != 'default':
                scheduler.step()
            step += 1
    
    golden = []
    predictions = []
    
    net.eval()
    for batch_idx, data in enumerate(testloader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = net(ids, mask, token_type_ids)

        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        golden.extend(data['targets'].tolist())
    return golden, predictions


def delayed_ensemble(dataset, trainloader, testloader, model_initialisation_seed, model_randomness_seed, noise_params):
    new_models_to_create = args.ensemble_size - 1
    net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed)
    net.cuda()
    if OPTIMISER == 'Adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    elif OPTIMISER == 'AdamW':
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE, weight_decay=0 if SCHEDULER != 'default' else 0.01)
    
    num_steps = NUM_EPOCHS * len(trainloader)
    if SCHEDULER != 'default':
        if 'warmup' in SCHEDULER:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
        else:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    net.train()

    step = 1
    train_mode = 'single'
    new_models = []
    new_optimisers = []
    for epoch in range(NUM_EPOCHS):
        for batch_idx, data in enumerate(trainloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            if step >= int(noise_params['start_ensemble'] * num_steps) and train_mode == 'single':
                print(step)
                print('Creating multiple models for the final part of training!')
                train_mode = 'final'
                states = get_rng_state()
                restore_rng_state(NOISE_PARAMS['states'])
                for idx in range(new_models_to_create):
                    new_net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed).cuda()
                    with torch.no_grad():
                        net_params = dict(net.named_parameters())
                        for name, param in new_net.named_parameters():
                            param.data = copy.deepcopy(net_params[name].data)
                            if (name in ['output.weight', 'output.bias'] and noise_params['noise_layer'] == 'head') or (noise_params['noise_layer'] == 'trainable' and param.requires_grad == True):
                                param.data += ((noise_params['noise_variance'])**0.5) * torch.randn(param.shape).cuda() * (torch.std(param.data) if noise_params['noise_type'] == 'scaled' else 1)
                    new_optimizer = OPTIMISER_MAPPER[OPTIMISER](params=new_net.parameters(), lr=LEARNING_RATE)
                    new_models.append(new_net)
                    new_optimisers.append(new_optimizer)
                NOISE_PARAMS['states'] = get_rng_state()
                restore_rng_state(states)

            if train_mode == 'single':
                optimizer.zero_grad()
                outputs = net(ids, mask, token_type_ids)
                loss = loss_fn(outputs, targets)

                loss.backward()
                optimizer.step()
                if SCHEDULER != 'default':
                    scheduler.step()
            else:
                optimizer.zero_grad()
                outputs = net(ids, mask, token_type_ids)
                loss = loss_fn(outputs, targets)

                loss.backward()
                optimizer.step()
                if SCHEDULER != 'default':
                    scheduler.step()

                for n_model, n_optimizer in zip(new_models, new_optimisers):
                    n_optimizer.zero_grad()
                    outputs = n_model(ids, mask, token_type_ids)
                    loss = loss_fn(outputs, targets)

                    loss.backward()
                    n_optimizer.step()
            step += 1
    
    golden = []
    predictions = []

    t_golden = []
    t_predictions = []
    net.eval()
    for batch_idx, data in enumerate(testloader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = net(ids, mask, token_type_ids)

        _, predicted = torch.max(outputs.data, 1)
        t_predictions.extend(predicted.tolist())
        t_golden.extend(data['targets'].tolist())
    golden.append(t_golden)
    predictions.append(t_predictions)

    for n_model in new_models:
        n_model.eval()
        t_golden = []
        t_predictions = []
        for batch_idx, data in enumerate(testloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            outputs = net(ids, mask, token_type_ids)

            _, predicted = torch.max(outputs.data, 1)
            t_predictions.extend(predicted.tolist())
            t_golden.extend(data['targets'].tolist())
        golden.append(t_golden)
        predictions.append(t_predictions)
    return golden, predictions


def noisy_interpolation(dataset, trainloader, testloader, model_initialisation_seed, model_randomness_seed, noise_params):
    new_models_to_create = args.ensemble_size - 1
    net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed)
    net.cuda()
    if OPTIMISER == 'Adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    elif OPTIMISER == 'AdamW':
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE, weight_decay=0 if SCHEDULER != 'default' else 0.01)
    
    num_steps = NUM_EPOCHS * len(trainloader)
    if SCHEDULER != 'default':
        if 'warmup' in SCHEDULER:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
        else:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    net.train()

    step = 1
    train_mode = 'single'
    new_models = []
    new_optimisers = []
    for epoch in range(NUM_EPOCHS):
        for batch_idx, data in enumerate(trainloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            if ((step >= int(noise_params['start_noise'] * num_steps) and step < int(noise_params['end_noise'] * num_steps)) or train_mode == 'multi') and (noise_params['step_type'] == 'step' or batch_idx == 0):
                if train_mode == 'single' and steps_since_last >= noise_params['noise_after_steps']:
                    print('Creating multiple models!')
                    steps_since_last = 0
                    train_mode = 'multi'
                    states = get_rng_state()
                    restore_rng_state(NOISE_PARAMS['states'])
                    for idx in range(new_models_to_create):
                        new_net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed).cuda()
                        with torch.no_grad():
                            net_params = dict(net.named_parameters())
                            for name, param in new_net.named_parameters():
                                param.data = copy.deepcopy(net_params[name].data)
                                if (name in ['output.weight', 'output.bias'] and noise_params['noise_layer'] == 'head') or (noise_params['noise_layer'] == 'trainable' and param.requires_grad == True):
                                    param.data += ((noise_params['noise_variance'] / step)**0.5) * torch.randn(param.shape).cuda() * (torch.std(param.data) if noise_params['noise_type'] == 'scaled' else 1)
                        
                        
                        new_optimizer = OPTIMISER_MAPPER[OPTIMISER](params=new_net.parameters(), lr=LEARNING_RATE)
                        new_models.append(new_net)
                        new_optimisers.append(new_optimizer)
                    NOISE_PARAMS['states'] = get_rng_state()
                    restore_rng_state(states)
                elif train_mode == 'multi' and steps_since_last >= noise_params['ensemble_training_steps']:
                    print('Aggregating multiple models into one!')
                    steps_since_last = 0
                    train_mode = 'single'
    
                    with torch.no_grad():
                        n_model_params = [dict(n_model.named_parameters()) for n_model in new_models]
                        for name, param in net.named_parameters():
                            for n_model_param in n_model_params:
                                param.data += n_model_param[name].data
                            param.data /= (1 + new_models_to_create)
                        
                        for n_model in new_models:
                            del n_model
                        del new_models
                        del new_optimisers
                        new_models = []
                        new_optimisers = []

            if train_mode == 'single':
                optimizer.zero_grad()
                outputs = net(ids, mask, token_type_ids)
                loss = loss_fn(outputs, targets)

                loss.backward()
                optimizer.step()
                if SCHEDULER != 'default':
                    scheduler.step()
            else:
                optimizer.zero_grad()
                outputs = net(ids, mask, token_type_ids)
                loss = loss_fn(outputs, targets)

                loss.backward()
                optimizer.step()
                if SCHEDULER != 'default':
                    scheduler.step()

                for n_model, n_optimizer in zip(new_models, new_optimisers):
                    n_optimizer.zero_grad()
                    outputs = n_model(ids, mask, token_type_ids)
                    loss = loss_fn(outputs, targets)

                    loss.backward()
                    n_optimizer.step()
            step += 1
            if noise_params['step_type'] == 'step':
                steps_since_last += 1
        if noise_params['step_type'] == 'epoch':
            steps_since_last += 1
    
    golden = []
    predictions = []

    net.eval()
    for batch_idx, data in enumerate(testloader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = net(ids, mask, token_type_ids)

        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        golden.extend(data['targets'].tolist())
    return golden, predictions


def delayed_ensemble_with_noisy_interpolation(dataset, trainloader, testloader, model_initialisation_seed, model_randomness_seed, noise_params):
    new_models_to_create = args.ensemble_size - 1
    net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed)
    net.cuda()
    if OPTIMISER == 'Adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    elif OPTIMISER == 'AdamW':
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE, weight_decay=0 if SCHEDULER != 'default' else 0.01)
    
    num_steps = NUM_EPOCHS * len(trainloader)
    if SCHEDULER != 'default':
        if 'warmup' in SCHEDULER:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * .1), num_steps)
        else:
            if 'cosine' in SCHEDULER:
                scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_steps)
            elif 'linear' in SCHEDULER:
                scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    net.train()

    step = 1
    train_mode = 'single'
    new_models = []
    new_optimisers = []
    for epoch in range(NUM_EPOCHS):
        for batch_idx, data in enumerate(trainloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            if ((step >= int(noise_params['start_noise'] * num_steps) and step < int(noise_params['end_noise'] * num_steps)) or train_mode == 'multi') and (noise_params['step_type'] == 'step' or batch_idx == 0):
                if train_mode == 'single' and steps_since_last >= noise_params['noise_after_steps']:
                    print('Creating multiple models!')
                    steps_since_last = 0
                    train_mode = 'multi'
                    states = get_rng_state()
                    restore_rng_state(NOISE_PARAMS['states'])
                    for idx in range(new_models_to_create):
                        new_net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed).cuda()
                        with torch.no_grad():
                            net_params = dict(net.named_parameters())
                            for name, param in new_net.named_parameters():
                                param.data = copy.deepcopy(net_params[name].data)
                                if (name in ['output.weight', 'output.bias'] and noise_params['noise_layer'] == 'head') or (noise_params['noise_layer'] == 'trainable' and param.requires_grad == True):
                                    param.data += ((noise_params['noise_variance'] / step)**0.5) * torch.randn(param.shape).cuda() * (torch.std(param.data) if noise_params['noise_type'] == 'scaled' else 1)
                        new_optimizer = OPTIMISER_MAPPER[OPTIMISER](params=new_net.parameters(), lr=LEARNING_RATE)
                        new_models.append(new_net)
                        new_optimisers.append(new_optimizer)
                    NOISE_PARAMS['states'] = get_rng_state()
                    restore_rng_state(states)
                elif train_mode == 'multi' and steps_since_last >= noise_params['ensemble_training_steps']:
                    print('Aggregating multiple models into one!')
                    steps_since_last = 0
                    train_mode = 'single'
    
                    with torch.no_grad():
                        n_model_params = [dict(n_model.named_parameters()) for n_model in new_models]
                        for name, param in net.named_parameters():
                            for n_model_param in n_model_params:
                                param.data += n_model_param[name].data
                            param.data /= (1 + new_models_to_create)
                        
                        for n_model in new_models:
                            del n_model
                        del new_models
                        del new_optimisers
                        new_models = []
                        new_optimisers = []
            elif step >= int(noise_params['start_ensemble'] * num_steps) and train_mode == 'single':
                print(step)
                print('Creating multiple models for the final part of training!')
                train_mode = 'final'
                states = get_rng_state()
                restore_rng_state(NOISE_PARAMS['states'])
                for idx in range(new_models_to_create):
                    new_net = get_ft_model(dataset.n_classes, model_initialisation_seed, model_randomness_seed).cuda()
                    with torch.no_grad():
                        net_params = dict(net.named_parameters())
                        for name, param in new_net.named_parameters():
                            param.data = copy.deepcopy(net_params[name].data)
                            if (name in ['output.weight', 'output.bias'] and noise_params['noise_layer'] == 'head') or (noise_params['noise_layer'] == 'trainable' and param.requires_grad == True):
                                param.data += ((noise_params['noise_variance'])**0.5) * torch.randn(param.shape).cuda() * (torch.std(param.data) if noise_params['noise_type'] == 'scaled' else 1)
                    new_optimizer = OPTIMISER_MAPPER[OPTIMISER](params=new_net.parameters(), lr=LEARNING_RATE)
                    new_models.append(new_net)
                    new_optimisers.append(new_optimizer)
                NOISE_PARAMS['states'] = get_rng_state()
                restore_rng_state(states)

            if train_mode == 'single':
                optimizer.zero_grad()
                outputs = net(ids, mask, token_type_ids)
                loss = loss_fn(outputs, targets)

                loss.backward()
                optimizer.step()
                if SCHEDULER != 'default':
                    scheduler.step()
            else:
                optimizer.zero_grad()
                outputs = net(ids, mask, token_type_ids)
                loss = loss_fn(outputs, targets)

                loss.backward()
                optimizer.step()
                if SCHEDULER != 'default':
                    scheduler.step()

                for n_model, n_optimizer in zip(new_models, new_optimisers):
                    n_optimizer.zero_grad()
                    outputs = n_model(ids, mask, token_type_ids)
                    loss = loss_fn(outputs, targets)

                    loss.backward()
                    n_optimizer.step()
            step += 1
            if noise_params['step_type'] == 'step':
                steps_since_last += 1
        if noise_params['step_type'] == 'epoch':
            steps_since_last += 1 
    golden = []
    predictions = []

    t_golden = []
    t_predictions = []
    net.eval()
    for batch_idx, data in enumerate(testloader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = net(ids, mask, token_type_ids)

        _, predicted = torch.max(outputs.data, 1)
        t_predictions.extend(predicted.tolist())
        t_golden.extend(data['targets'].tolist())
    golden.append(t_golden)
    predictions.append(t_predictions)

    for n_model in new_models:
        n_model.eval()
        t_golden = []
        t_predictions = []
        for batch_idx, data in enumerate(testloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            outputs = net(ids, mask, token_type_ids)

            _, predicted = torch.max(outputs.data, 1)
            t_predictions.extend(predicted.tolist())
            t_golden.extend(data['targets'].tolist())
        golden.append(t_golden)
        predictions.append(t_predictions)
    return golden, predictions


def get_ft_model(n_classes, model_initialisation_seed, model_randomness_seed):
    if PEFT is None:
        print('Creating basic finetuning model')
        return FT_MODELS[MODEL](n_classes, model_initialisation_seed, model_randomness_seed, True)
    else:
        print(f'Creating peft finetuning model with {PEFT}')
        return FT_MODELS[f'peft_{MODEL}'](n_classes, model_initialisation_seed, model_randomness_seed, False, PEFT)        

def ft_experiment(randomness_factor_seeds):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, return_dict=False)
    dataset = FineTuningDataset(
        dataset_name=DATASET,
        train_size=args.train_size,
        num_labelled=args.num_labelled,
        num_labelled_test=args.num_labelled_test,
        split_seed=randomness_factor_seeds['data_split'],
        label_seed=randomness_factor_seeds['label_choice'],
        device=device,
        full_test=FULL_TEST,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        augmented_data_size=AUGMENTED,
    )
    loader = DatasetLoader(DATASET, BATCH_SIZE, dataset, randomness_factor_seeds['sample_order'])
    trainloader = loader.trainloader()
    testloader = loader.testloader()
    output_preds = []
    old_states = set_rng_state(NOISE_PARAMS['seed'])
    NOISE_PARAMS['states'] = get_rng_state()
    restore_rng_state(old_states)
    if OPTIMISATION_MITIGATION == 'default':
        golden, predictions, output_preds = default_experiment(dataset, trainloader, testloader, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'])
    elif OPTIMISATION_MITIGATION == 'ensemble':
        random.seed(randomness_factor_seeds['model_initialisation'])
        init_seeds = [random.randint(1, 100000) for _ in range(args.ensemble_size)]
        random.seed(randomness_factor_seeds['model_randomness'])
        randomness_seeds = [random.randint(1, 100000) for _ in range(args.ensemble_size)]

        golden = []
        predictions = []
        for ensemble_idx in range(args.ensemble_size):
            g, p, o = default_experiment(dataset, trainloader, testloader, init_seeds[ensemble_idx], randomness_seeds[ensemble_idx])
            golden.append(g)
            predictions.append(p)
            output_preds.append(o)
    elif OPTIMISATION_MITIGATION == 'noise':
        golden, predictions = noise_regularisation_experiment(dataset, trainloader, testloader, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'], NOISE_PARAMS)
    elif OPTIMISATION_MITIGATION == 'prior_noise':
        golden, predictions = prior_noise_regularisation_experiment(dataset, trainloader, testloader, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'], NOISE_PARAMS)
    
    elif OPTIMISATION_MITIGATION in ['delayed_ensemble', 'de']:
        golden, predictions = delayed_ensemble(dataset, trainloader, testloader, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'], NOISE_PARAMS)
    elif OPTIMISATION_MITIGATION in ['noisy_interpolation', 'ni']:
        golden, predictions = noisy_interpolation(dataset, trainloader, testloader, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'], NOISE_PARAMS)
    elif OPTIMISATION_MITIGATION in ['delayed_ensemble_with_noisy_interpolation', 'deni']:
        golden, predictions = delayed_ensemble_with_noisy_interpolation(dataset, trainloader, testloader, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'], NOISE_PARAMS)
    
    elif OPTIMISATION_MITIGATION == 'swa':
        golden, predictions = SWA_ft_experiment(dataset, trainloader, testloader, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'])
    elif OPTIMISATION_MITIGATION == 'ema':
        golden, predictions = EMA_ft_experiment(dataset, trainloader, testloader, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'])
    elif OPTIMISATION_MITIGATION == 'mixout':
        golden, predictions = mixout_experiment(dataset, trainloader, testloader, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'])

    return golden, predictions, output_preds

parser = argparse.ArgumentParser()
# Meta
parser.add_argument('--experiment_name', default='investigation_experiments', type=str, help='Directory to save experiments to')
parser.add_argument('--configuration_name', default='stability', type=str, help='Further distinction for the save directory')
parser.add_argument('--experiment_type', default='finetuning', type=str, help='Type of experiment to run')
parser.add_argument('--full_test', default=1, type=int, help='Whether to use whole test dataset (Yes (default): 1; No: 0). If "No" and "num_labelled_test" is not set then uses same number of labelled samples as defined by num_labelled')
parser.add_argument('--regenerate', default=0, type=int, help='Whether to calculate every result again or continue from checkpoint (Yes: 1; No (default): 0).')
# General training args
parser.add_argument('--factor', default='golden_model', type=str, choices=['golden_model', 'data_split', 'label_choice', 'sample_choice', 'sample_order', 'model_initialisation', 'model_randomness', 'optimisation'], help='Randomness factor to investigate.')
parser.add_argument('--dataset', default='sst2', type=str, choices=['sst2', 'mrpc', 'cola', 'rte', 'boolq', 'trec', 'ag_news', 'db_pedia', 'snips'], help='Dataset to use for investigation.')
parser.add_argument('--num_classes', default=2, type=int, help='Number of classes in dataset.')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--train_size', default=0.8, type=float)
parser.add_argument('--num_labelled', default=1000, type=int)
parser.add_argument('--num_labelled_test', default=1000, type=int)
parser.add_argument('--model', default='flan-t5', type=str, choices=['bert', 'roberta', 'albert'])
parser.add_argument('--model_size', default='base', type=str, choices=['base'])
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--num_epochs', default=5, type=int, help='Total number of epochs to train for')
parser.add_argument('--max_len', default=20, type=int, help='Maximal length of input for fine-tuning experiments')
parser.add_argument('--augmented_data_size', default=0, type=int, help='How many augmented samples to use per available samples. Option -1 indicated that all augmented samples from data directory are used for the specific dataset (defaulted to 10)')
# Transfer learning
parser.add_argument('--optimizer', default='Adam', type=str, choices=['Adam', 'AdamW'], help='What optimiser to use')
parser.add_argument('--scheduler', default='linear', type=str, choices=['linear', 'cosine', 'linear_warmup', 'cosine_warmup'], help='What scheduler to use')
parser.add_argument('--optimisation_mitigation', default='default', type=str, choices=['default', 'ensemble', 'noise', 'prior_noise', 'swa', 'ema', 'mixout', 'delayed_ensemble', 'de', 'noisy_interpolation', 'ni', 'delayed_ensemble_with_noisy_interpolation', 'deni'], help='What optimisation mitigation to use')
parser.add_argument('--peft', default=None, type=str, choices=[None, 'lora', 'ia3', 'prompt_tuning', 'unipelt'], help='What PEFT to use. If None then no PEFT is used')
parser.add_argument('--ensemble_size', default=10, type=int)
parser.add_argument('--noise_location', default='weights', type=str, choices=['input', 'weights'], help='Where to add noise')
parser.add_argument('--noise_variance', default=0.1, type=float, help='The size of noise to add. The noise will be Gaussian noise with mean of 0 and the variance defined by this parameter.')
parser.add_argument('--noise_type', default='uniform', type=str, choices=['uniform', 'scaled'], help='Add uniform noise or scale based on deviation of weights?')
parser.add_argument('--noise_layer', default='head', type=str, choices=['head', 'trainable'], help='Add noise only to the classification head or to all trainable layers?')

parser.add_argument('--start_noise', default=.2, type=float, help='The percentage of steps after which to start adding noise.')
parser.add_argument('--end_noise', default=.6, type=float, help='The percentage of steps after which to end adding noise.')
parser.add_argument('--start_ensemble', default=.8, type=float, help='The percentage of steps after which to create the final ensemble.')
parser.add_argument('--step_type', default='epoch', type=str, choices=['step', 'epoch'], help='Whether the `noise_after_steps` and `ensemble_training_steps` are counting steps or epochs.')
parser.add_argument('--noise_after_steps', default=5, type=int, help='The number of steps after which the noise is added (for basic noise regularisation) or the ensemble is created (for our method in conitnuous or combined setting).')
parser.add_argument('--ensemble_training_steps', default=5, type=int, help='The number of steps to run, before next noise is added (or the the model are averaged for our method).')

# Seeds
parser.add_argument('--mitigation_seed', default=42, type=int, help='Seed for generating seeds for investigation')
parser.add_argument('--investigation_seed', default=27, type=int, help='Seed for generating seeds for investigation')
# Investigation
parser.add_argument('--investigation_runs', default=10, type=int, help='Number of different configurations for investigating chosen randomness factor.')
parser.add_argument('--mitigation_runs', default=100, type=int, help='Number of different configurations for mitigating other randomness factors.')

parser.add_argument('-f')
args = parser.parse_args()

device = torch.device('cuda')
FT_MODELS = {
    'bert':  BERTBase,
    'roberta':  RoBERTaBase,
    'albert': ALBERTBase,
    'peft_bert':  PEFTBERTBase,
    'peft_roberta':  PEFTRoBERTaBase,
    'peft_albert': PEFTALBERTBase,
}

OPTIMISER_MAPPER = {
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
}

EXPERIMENT_TYPE = args.experiment_type
FULL_TEST = args.full_test == 1
MAX_LEN = args.max_len
PEFT = args.peft
AUGMENTED = args.augmented_data_size

OPTIMISER = args.optimizer
SCHEDULER = args.scheduler
OPTIMISATION_MITIGATION = args.optimisation_mitigation

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic

NOISE_PARAMS = {k: getattr(args, k) for k in ['noise_location', 'noise_variance', 'noise_after_steps', 'noise_type', 'noise_layer', 'start_noise', 'end_noise', 'start_ensemble', 'ensemble_training_steps', 'step_type']}    
print(NOISE_PARAMS)

MODEL = args.model
MODEL_SIZE = args.model_size
FACTOR = args.factor
DATASET = args.dataset
print(f'Running investigation for dataset {DATASET}')
print(args.optimisation_mitigation)
if PEFT is None:
    print(f'Running model {MODEL}')
    RESULTS_PATH = os.path.join('results', f'{args.experiment_name}', f'{EXPERIMENT_TYPE}_{MODEL}_{MODEL_SIZE}', args.configuration_name, DATASET, FACTOR)
else:
    print(f'Running PEFT method {PEFT} on model {MODEL}')
    RESULTS_PATH = os.path.join('results', f'{args.experiment_name}', f'{EXPERIMENT_TYPE}_{PEFT}_{MODEL}_{MODEL_SIZE}', args.configuration_name, DATASET, FACTOR)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

BATCH_SIZE = args.batch_size # 64
NUM_EPOCHS = args.num_epochs # 5
LEARNING_RATE = args.lr # 1e-5

if os.path.exists(os.path.join(RESULTS_PATH, 'mitigation_seeds.pkl')):
    with open(os.path.join(RESULTS_PATH, 'mitigation_seeds.pkl'), 'rb') as file:
        print(f'Loading mitigation seeds:')
        mitigation_seeds = pickle.load(file)
        print(mitigation_seeds)
        print(f'Length of mitigation seeds: {len(mitigation_seeds)}')
if not os.path.exists(os.path.join(RESULTS_PATH, 'mitigation_seeds.pkl')) or len(mitigation_seeds) != args.mitigation_runs:
    print(f'Generating new mitigation seeds of length: {args.mitigation_runs}')
    random.seed(args.mitigation_seed)
    mitigation_seeds = [random.randint(1, 100000) for _ in range(args.mitigation_runs)]
    print(mitigation_seeds)
    with open(os.path.join(RESULTS_PATH, 'mitigation_seeds.pkl'), 'wb') as file:
        pickle.dump(mitigation_seeds, file)


if os.path.exists(os.path.join(RESULTS_PATH, 'investigation_seeds.pkl')):
    with open(os.path.join(RESULTS_PATH, 'investigation_seeds.pkl'), 'rb') as file:
        print(f'Loading investigation seeds:')
        investigation_seeds = pickle.load(file)
        print(investigation_seeds)
        print(f'Length of investigation seeds: {len(investigation_seeds)}')
if not os.path.exists(os.path.join(RESULTS_PATH, 'investigation_seeds.pkl')) or len(investigation_seeds) != args.investigation_runs:
    print(f'Generating new investigation seeds of length: {args.investigation_runs}')
    random.seed(args.investigation_seed)
    investigation_seeds = [random.randint(1, 100000) for _ in range(args.investigation_runs)]   
    print(investigation_seeds)
    with open(os.path.join(RESULTS_PATH, 'investigation_seeds.pkl'), 'wb') as file:
        pickle.dump(investigation_seeds, file)


if MODEL == 'bert':
    suffix = '-uncased'
elif MODEL == 'albert':
    suffix = '-v2'
else:
    suffix = ''
MODEL_NAME = f'{MODEL}-{MODEL_SIZE}{suffix}'


randomness_factors = ['data_split', 'label_choice', 'sample_choice', 'sample_order', 'model_initialisation', 'model_randomness']

print(f'Running investigation for factor {FACTOR}')

for mit_idx, mitigation_seed in enumerate(mitigation_seeds):
    print(f'Running mitigation number {mit_idx} with seed {mitigation_seed}')
    mitigation_path = os.path.join(RESULTS_PATH, f'mitigation_{mit_idx}')
    if not os.path.exists(mitigation_path):
        os.makedirs(mitigation_path)
    
    randomness_factor_seeds = {factor: mitigation_seed for factor in randomness_factors}

    for inv_idx, investigation_seed in enumerate(investigation_seeds):
        print(f'Running investigation number {inv_idx} with seed {investigation_seed}')
        investigation_path = os.path.join(mitigation_path, f'investigation_{inv_idx}')
        if os.path.exists(os.path.join(investigation_path, 'results.json')) and args.regenerate == 0:
            print(f'Investigation number {inv_idx} already exists under mitigation {mit_idx}. Skipping!')
            continue
        if not os.path.exists(investigation_path):
            os.makedirs(investigation_path)
        
        if FACTOR != 'golden_model':
            if FACTOR == 'optimisation':
                for factor in randomness_factors:
                    if factor not in ['data_split', 'label_choice']:
                        randomness_factor_seeds[factor] = investigation_seed
                NOISE_PARAMS['seed'] = investigation_seed
            else:
                randomness_factor_seeds[FACTOR] = investigation_seed

        output_preds = []
        print(f'Running fine-tuning experiments!')
        golden, predicted, output_preds = ft_experiment(randomness_factor_seeds)
        
        print(np.mean(np.array(golden) == np.array(predicted)))
        results = copy.deepcopy(randomness_factor_seeds)
        results['real'] = golden
        results['predicted'] = predicted
        results['base_model'] = MODEL_NAME
        results['mitigation_idx'] = mit_idx
        results['investigation_idx'] = inv_idx
        results['output_preds'] = output_preds

        with open(os.path.join(investigation_path, 'results.json'), 'w') as file:
            json.dump(results, file)

