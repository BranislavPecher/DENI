import torch
import random
import numpy as np
from transformers import BertModel, RobertaModel, AlbertModel, BertForSequenceClassification, BertConfig, RobertaConfig, RobertaForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model, IA3Model, IA3Config, PromptTuningConfig, PromptTuningInit
from adapters import AutoAdapterModel, AdapterConfig, BertAdapterModel, ConfigUnion, LoRAConfig, PrefixTuningConfig, SeqBnConfig, init

class DeterministicModel():
    def __init__(self):
        old_torch_state = torch.get_rng_state()
        old_torch_cuda_state = torch.cuda.get_rng_state()
        old_numpy_state = np.random.get_state()
        old_random_state = random.getstate()

    def set_rng_state(self, seed):
        old_torch_state = torch.get_rng_state()
        old_torch_cuda_state = torch.cuda.get_rng_state()
        old_numpy_state = np.random.get_state()
        old_random_state = random.getstate()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        return old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state

    def restore_rng_state(self, states):
        old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state = states

        torch.set_rng_state(old_torch_state)
        torch.cuda.set_rng_state(old_torch_cuda_state)
        np.random.set_state(old_numpy_state)
        random.setstate(old_random_state)

    def get_rng_state(self):
        old_torch_state = torch.get_rng_state()
        old_torch_cuda_state = torch.cuda.get_rng_state()
        old_numpy_state = np.random.get_state()
        old_random_state = random.getstate()
        return old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state


class BERTBase(torch.nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=True):
        self.name = 'bert-base'
        states = self.set_rng_state(init_seed)
        super(BERTBase, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        if not trainable:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.restore_rng_state(states)

        states = self.set_rng_state(dropout_seed)
        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

    def forward(self, input_ids, attention_mask, token_type_ids, add_noise=None):
        states = self.get_rng_state()
        self.restore_rng_state(self.dropout_states)

        if add_noise is not None:
            with torch.no_grad():
                _, bert_output = self.bert(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  token_type_ids=token_type_ids
                )
                bert_output += (add_noise**0.5) * torch.randn(bert_output.shape).cuda()
        else:
            _, bert_output = self.bert(
              input_ids=input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
            )
        
        output = self.dropout(bert_output)
        output = self.output(output)

        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

        return output

class RoBERTaBase(torch.nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=True):
        self.name = 'roberta-base'
        states = self.set_rng_state(init_seed)
        super(RoBERTaBase, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base', return_dict=None)
        if not trainable:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.restore_rng_state(states)

        states = self.set_rng_state(dropout_seed)
        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

    def forward(self, input_ids, attention_mask, token_type_ids, add_noise=None):
        states = self.get_rng_state()
        self.restore_rng_state(self.dropout_states)

        if add_noise is not None:
            with torch.no_grad():
                _, bert_output = self.bert(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  token_type_ids=token_type_ids
                )
                bert_output += (add_noise**0.5) * torch.randn(bert_output.shape).cuda()
        else:
            _, bert_output = self.bert(
              input_ids=input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
            )
        
        output = self.dropout(bert_output)
        output = self.output(output)

        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

        return output

class ALBERTBase(torch.nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=True):
        self.name = 'albert-base'
        states = self.set_rng_state(init_seed)
        super(ALBERTBase, self).__init__()
        self.bert = AlbertModel.from_pretrained('albert-base-v2', return_dict=False)
        if not trainable:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.restore_rng_state(states)

        states = self.set_rng_state(dropout_seed)
        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

    def forward(self, input_ids, attention_mask, token_type_ids, add_noise=None):
        states = self.get_rng_state()
        self.restore_rng_state(self.dropout_states)

        if add_noise is not None:
            with torch.no_grad():
                _, bert_output = self.bert(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  token_type_ids=token_type_ids
                )
                bert_output += (add_noise**0.5) * torch.randn(bert_output.shape).cuda()
        else:
            _, bert_output = self.bert(
              input_ids=input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
            )
        
        output = self.dropout(bert_output)
        output = self.output(output)

        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

        return output

class PEFTBERTBase(torch.nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=False, peft='lora', task_text=None):
        self.peft = peft
        self.task_text = task_text
        self.name = 'bert-base'
        states = self.set_rng_state(init_seed)
        super(PEFTBERTBase, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        if not trainable:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.__configure_peft_model__()
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        
        self.restore_rng_state(states)

        states = self.set_rng_state(dropout_seed)
        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

    def __configure_peft_model__(self):
        if self.peft not in ['unipelt']:
            if self.peft == 'lora':
                print('running lora')
                config = LoraConfig(r=64, lora_alpha=64, lora_dropout=0.1, task_type=TaskType.FEATURE_EXTRACTION, bias='all', use_rslora=True)
            elif self.peft == 'ia3':
                print('running ia3')
                config = IA3Config(peft_type="IA3", task_type=TaskType.FEATURE_EXTRACTION)
            elif self.peft == 'prompt_tuning':
                print('running prompt_tuning')
                config = PromptTuningConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    num_virtual_tokens=25,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    tokenizer_name_or_path='bert-base-uncased',
                    prompt_tuning_init_text=self.task_text,
                )
            else:
                raise NotImplementedError
            self.bert = get_peft_model(self.bert, config)
            print(self.bert.print_trainable_parameters())
        else:
            if self.peft == 'unipelt':
                print('running unipelt')
                init(self.bert)
                config = ConfigUnion(
                    LoRAConfig(r=64, alpha=64, dropout=0.1, use_gating=True),
                    PrefixTuningConfig(prefix_length=25, use_gating=True),
                    SeqBnConfig(reduction_factor=16, use_gating=True),
                )
                self.bert.add_adapter("unipelt", config=config)
                self.bert.set_active_adapters('unipelt')
                self.bert.train_adapter('unipelt')
            else:
                raise NotImplementedError
            trainable_params = 0
            all_param = 0
            for n, param in self.bert.named_parameters():
                num_params = param.numel()
                if num_params == 0 and hasattr(param, "ds_numel"):
                    num_params = param.ds_numel
            
                if param.__class__.__name__ == "Params4bit":
                    num_params = num_params * 2
            
                all_param += num_params
                if param.requires_grad:
                    # print(n)
                    trainable_params += num_params
            print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")

    def forward(self, input_ids, attention_mask, token_type_ids, add_noise=None):
        states = self.get_rng_state()
        self.restore_rng_state(self.dropout_states)

        if add_noise is not None:
            with torch.no_grad():
                _, bert_output = self.bert(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  token_type_ids=token_type_ids
                )
                bert_output += (add_noise**0.5) * torch.randn(bert_output.shape).cuda()
        else:
            _, bert_output = self.bert(
              input_ids=input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
            )
      
        output = self.dropout(bert_output)
        output = self.output(output)

        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

        return output

class PEFTRoBERTaBase(torch.nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=False, peft='lora', task_text=None):
        self.peft = peft
        self.task_text = task_text
        self.name = 'roberta-base'
        states = self.set_rng_state(init_seed)
        super(PEFTRoBERTaBase, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base', return_dict=None)
        if not trainable:
            for name, param in self.bert.named_parameters():
                if name not in ['pooler.dense.bias', 'pooler.dense.weight']:
                    param.requires_grad = False
        self.__configure_peft_model__()
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        
        self.restore_rng_state(states)

        states = self.set_rng_state(dropout_seed)
        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

    def __configure_peft_model__(self):
        if self.peft not in ['unipelt']:
            if self.peft == 'lora':
                print('running lora')
                modules_to_target = []
                for name, module in self.bert.named_modules():
                    if ('dense' in name or 'query' in name or 'key' in name or 'value' in name) and 'pooler' not in name:
                        modules_to_target.append(name)
                config = LoraConfig(r=64, lora_alpha=64, lora_dropout=0.1, task_type=TaskType.FEATURE_EXTRACTION, bias='all', use_rslora=True, target_modules=modules_to_target)
            elif self.peft == 'ia3':
                print('running ia3')
                modules_to_target = []
                linear_modules = []
                for name, module in self.bert.named_modules():
                    if ('query' in name or 'key' in name or 'value' in name) and 'pooler' not in name:
                        modules_to_target.append(name)
                    if 'dense' in name and 'pooler' not in name:
                        modules_to_target.append(name)
                        linear_modules.append(name)
                    
                config = IA3Config(peft_type="IA3", task_type=TaskType.FEATURE_EXTRACTION, target_modules=modules_to_target, feedforward_modules=linear_modules)
            elif self.peft == 'prompt_tuning':
                print('running prompt_tuning')
                config = PromptTuningConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    num_virtual_tokens=25,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    tokenizer_name_or_path='roberta-base',
                    prompt_tuning_init_text=self.task_text,
                )
            else:
                raise NotImplementedError
            self.bert = get_peft_model(self.bert, config)
            for name, param in self.bert.named_parameters():
            # print(name)
                if name in ['base_model.model.pooler.dense.bias', 'base_model.model.pooler.dense.weight', 'base_model.pooler.dense.bias', 'base_model.pooler.dense.weight']:
                    print(name)
                    param.requires_grad = True
            print(self.bert.print_trainable_parameters())
        else:
            if self.peft == 'unipelt':
                print('running unipelt')
                init(self.bert)
                config = ConfigUnion(
                    LoRAConfig(r=64, alpha=64, dropout=0.1, use_gating=True),
                    PrefixTuningConfig(prefix_length=25, use_gating=True),
                    SeqBnConfig(reduction_factor=16, use_gating=True),
                )
                self.bert.add_adapter("unipelt", config=config)
                self.bert.set_active_adapters('unipelt')
                self.bert.train_adapter('unipelt')
            else:
                raise NotImplementedError
            for name, param in self.bert.named_parameters():
                # print(name)
                if name in ['base_model.model.pooler.dense.bias', 'base_model.model.pooler.dense.weight', 'base_model.pooler.dense.bias', 'base_model.pooler.dense.weight', 'pooler.dense.bias', 'pooler.dense.weight']:
                    print(name)
                    param.requires_grad = True
            
            trainable_params = 0
            all_param = 0
            for n, param in self.bert.named_parameters():
                num_params = param.numel()
                if num_params == 0 and hasattr(param, "ds_numel"):
                    num_params = param.ds_numel
            
                if param.__class__.__name__ == "Params4bit":
                    num_params = num_params * 2
            
                all_param += num_params
                if param.requires_grad:
                    # print(n)
                    trainable_params += num_params
            print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")

    def forward(self, input_ids, attention_mask, token_type_ids, add_noise=None):
        states = self.get_rng_state()
        self.restore_rng_state(self.dropout_states)

        if add_noise is not None:
            with torch.no_grad():
                _, bert_output = self.bert(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  token_type_ids=token_type_ids
                )
                bert_output += (add_noise**0.5) * torch.randn(bert_output.shape).cuda()
        else:
            _, bert_output = self.bert(
              input_ids=input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
            )
      
        output = self.dropout(bert_output)
        output = self.output(output)

        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

        return output

class PEFTALBERTBase(torch.nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=False, peft='lora', task_text=None):
        self.peft = peft
        self.task_text = task_text
        self.name = 'albert-base'
        states = self.set_rng_state(init_seed)
        super(PEFTALBERTBase, self).__init__()
        self.bert = AlbertModel.from_pretrained('albert-base-v2', return_dict=False)
        if not trainable:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.__configure_peft_model__()
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        
        self.restore_rng_state(states)

        states = self.set_rng_state(dropout_seed)
        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

    def __configure_peft_model__(self):
        if self.peft not in ['unipelt']:
            if self.peft == 'lora':
                print('running lora')
                modules_to_target = []
                for name, module in self.bert.named_modules():
                    if ('key' in name or 'value' in name):
                        modules_to_target.append(name)
                config = LoraConfig(r=64, lora_alpha=64, lora_dropout=0.1, task_type=TaskType.FEATURE_EXTRACTION, bias='all', use_rslora=True, target_modules=modules_to_target)
            elif self.peft == 'ia3':
                print('running ia3')
                modules_to_target = []
                linear_modules = []
                for name, module in self.bert.named_modules():
                    if ('query' in name or 'value' in name or 'key' in name):
                        modules_to_target.append(name)
                    if 'ffn_output' in name or 'dense' in name:
                        modules_to_target.append(name)
                        linear_modules.append(name)
                config = IA3Config(peft_type="IA3", task_type=TaskType.FEATURE_EXTRACTION, target_modules=modules_to_target, feedforward_modules=linear_modules)
            elif self.peft == 'prompt_tuning':
                print('running prompt_tuning')
                config = PromptTuningConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    num_virtual_tokens=25,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    tokenizer_name_or_path='albert-base-v2',
                    prompt_tuning_init_text=self.task_text,
                )
            else:
                raise NotImplementedError
            self.bert = get_peft_model(self.bert, config)
            print(self.bert.print_trainable_parameters())
        else:
            if self.peft == 'unipelt':
                print('running unipelt')
                init(self.bert)
                config = ConfigUnion(
                    LoRAConfig(r=64, alpha=64, dropout=0.1, use_gating=True),
                    PrefixTuningConfig(prefix_length=25, use_gating=True),
                    SeqBnConfig(reduction_factor=16, use_gating=True),
                )
                self.bert.add_adapter("unipelt", config=config)
                self.bert.set_active_adapters('unipelt')
                self.bert.train_adapter('unipelt')
            else:
                raise NotImplementedError         
            
            trainable_params = 0
            all_param = 0
            for n, param in self.bert.named_parameters():
                num_params = param.numel()
                if num_params == 0 and hasattr(param, "ds_numel"):
                    num_params = param.ds_numel
            
                if param.__class__.__name__ == "Params4bit":
                    num_params = num_params * 2
            
                all_param += num_params
                if param.requires_grad:
                    # print(n)
                    trainable_params += num_params
            print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")

    def forward(self, input_ids, attention_mask, token_type_ids, add_noise=None):
        states = self.get_rng_state()
        self.restore_rng_state(self.dropout_states)

        if add_noise is not None:
            with torch.no_grad():
                _, bert_output = self.bert(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  token_type_ids=token_type_ids
                )
                bert_output += (add_noise**0.5) * torch.randn(bert_output.shape).cuda()
        else:
            _, bert_output = self.bert(
              input_ids=input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
            )
      
        output = self.dropout(bert_output)
        output = self.output(output)

        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

        return output