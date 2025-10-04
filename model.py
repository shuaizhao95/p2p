
import math
import torch
import torch.nn as nn


class LoRAExpertLayer(nn.Module):
    
    def __init__(self, in_features, out_features, num_experts=10, expert_rank=4, expert_alpha=2.0, dropout=0.1, task_type="detection"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.expert_rank = expert_rank
        self.expert_alpha = expert_alpha
        self.task_type = task_type  
        
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if i == 0:
                expert = nn.ModuleDict({
                    'lora_A': nn.Linear(in_features, expert_rank, bias=False),
                    'lora_B': nn.Linear(expert_rank, out_features, bias=False)
                })
            else:
                expert = nn.ModuleDict({
                    'lora_A': nn.Linear(in_features, expert_rank, bias=False),
                    'lora_B': nn.Linear(expert_rank, out_features, bias=False)
                })
            self.experts.append(expert)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for expert in self.experts:
            nn.init.kaiming_uniform_(expert['lora_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(expert['lora_B'].weight)
    
    def forward(self, x, expert_idx=None):
        if expert_idx is None:
            expert_idx = torch.randint(0, self.num_experts, (1,)).item()
        
        expert = self.experts[expert_idx]
        a_out = expert['lora_A'](x)
        a_out = self.dropout(a_out)
        b_out = expert['lora_B'](a_out)
        
        return b_out * (self.expert_alpha / self.expert_rank)


class MultiTaskLoRAModel(nn.Module):
    
    def __init__(self, base_model, target_modules, num_experts=10, expert_rank=4, expert_alpha=2.0, dropout=0.1):
        super().__init__()
        self.base_model = base_model
        self.target_modules = target_modules
        self.num_experts = num_experts
        self.expert_rank = expert_rank
        self.expert_alpha = expert_alpha
        self.dropout = dropout
        
    
        self.detection_experts = {}  
        self.classification_experts = {}  
        

        self.active_task = "detection"
        

        self._create_expert_layers()
        
    def _create_expert_layers(self):

        target_module_info = []
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    target_module_info.append((name, module))
        
        print(f"Total target modules for LoRA experts: {len(target_module_info)}")
        

        for name, module in target_module_info:
            in_features = module.in_features
            out_features = module.out_features
            
            detection_expert = LoRAExpertLayer(
                in_features=in_features,
                out_features=out_features,
                num_experts=self.num_experts,
                expert_rank=self.expert_rank,
                expert_alpha=self.expert_alpha,
                dropout=self.dropout,
                task_type="detection"
            )
            
            classification_expert = LoRAExpertLayer(
                in_features=in_features,
                out_features=out_features,
                num_experts=self.num_experts,
                expert_rank=self.expert_rank,
                expert_alpha=self.expert_alpha,
                dropout=self.dropout,
                task_type="classification"
            )
            
            device = next(self.base_model.parameters()).device
            dtype = next(self.base_model.parameters()).dtype
            detection_expert = detection_expert.to(device=device, dtype=dtype)
            classification_expert = classification_expert.to(device=device, dtype=dtype)
            
            self.detection_experts[name] = detection_expert
            self.classification_experts[name] = classification_expert
    
    def set_active_task(self, task_type, verbose=False):
        if task_type in ["detection", "classification"]:
            if self.active_task != task_type:
                self.active_task = task_type
                if verbose:
                    print(f"Switched to {task_type} task")
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def get_active_experts(self):
        if self.active_task == "detection":
            return self.detection_experts
        else:
            return self.classification_experts
    
    def forward(self, input_ids, attention_mask, task_type=None):
        if task_type is not None:
            self.set_active_task(task_type)
        
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        active_experts = self.get_active_experts()

        
        return base_output
    
    def get_lora_parameters(self, task_type=None):
        if task_type is None:
            task_type = self.active_task
        
        if task_type == "detection":
            experts = self.detection_experts
        else:
            experts = self.classification_experts
        
        lora_params = []
        for expert in experts.values():
            for param in expert.parameters():
                lora_params.append(param)
        
        return lora_params


def create_multi_task_lora_model(base_model, target_modules, num_experts=10, expert_rank=4, expert_alpha=2.0, dropout=0.1):
    return MultiTaskLoRAModel(
        base_model=base_model,
        target_modules=target_modules,
        num_experts=num_experts,
        expert_rank=expert_rank,
        expert_alpha=expert_alpha,
        dropout=dropout
    )


def create_lora_expert_model(base_model, target_modules, num_experts=10, expert_rank=4, expert_alpha=2.0, dropout=0.1):
    expert_layers = {}
    
    target_module_info = []
    for name, module in base_model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                target_module_info.append((name, module))
    
    print(f"Total target modules for LoRA experts: {len(target_module_info)}")
    
    for name, module in target_module_info:
        in_features = module.in_features
        out_features = module.out_features
        
        expert_layer = LoRAExpertLayer(
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            expert_rank=expert_rank,
            expert_alpha=expert_alpha,
            dropout=dropout
        )
        
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        expert_layer = expert_layer.to(device=device, dtype=dtype)
        
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        expert_layer_name = f"{child_name}_experts"
        
        if parent_name:
            parent_module = base_model
            for attr in parent_name.split('.'):
                parent_module = getattr(parent_module, attr)
            setattr(parent_module, expert_layer_name, expert_layer)
        else:
            setattr(base_model, expert_layer_name, expert_layer)
        
        expert_layers[name] = expert_layer
    
    return base_model, expert_layers


def apply_multi_task_lora_hooks(model, multi_task_model):
    
    def create_detection_hook(expert_layer):
        def hook(module, input, output):
            if len(input) > 0 and multi_task_model.active_task == "detection":
                x = input[0]
                if x.dtype != expert_layer.experts[0]['lora_A'].weight.dtype:
                    x = x.to(dtype=expert_layer.experts[0]['lora_A'].weight.dtype)
                
                lora_output = expert_layer.forward(x)
                
                if lora_output.dtype != output.dtype:
                    lora_output = lora_output.to(dtype=output.dtype)
                return output + lora_output
            return output
        return hook
    
    def create_classification_hook(expert_layer):
        def hook(module, input, output):
            if len(input) > 0 and multi_task_model.active_task == "classification":
                x = input[0]
                if x.dtype != expert_layer.experts[0]['lora_A'].weight.dtype:
                    x = x.to(dtype=expert_layer.experts[0]['lora_A'].weight.dtype)
                
                lora_output = expert_layer.forward(x)
                
                if lora_output.dtype != output.dtype:
                    lora_output = lora_output.to(dtype=output.dtype)
                return output + lora_output
            return output
        return hook
    
    hooks = []
    
    for name, module in model.named_modules():
        if name in multi_task_model.detection_experts:
            expert_layer = multi_task_model.detection_experts[name]
            hook = module.register_forward_hook(create_detection_hook(expert_layer))
            hooks.append(hook)
    
    for name, module in model.named_modules():
        if name in multi_task_model.classification_experts:
            expert_layer = multi_task_model.classification_experts[name]
            hook = module.register_forward_hook(create_classification_hook(expert_layer))
            hooks.append(hook)
    
    return hooks


def apply_lora_expert_forward_hook(model, expert_layers):

    def create_hook(expert_layer):
        def hook(module, input, output):
            if len(input) > 0:
                x = input[0]
                if x.dtype != expert_layer.experts[0]['lora_A'].weight.dtype:
                    x = x.to(dtype=expert_layer.experts[0]['lora_A'].weight.dtype)
                
                lora_output = expert_layer.forward(x)
                
                if lora_output.dtype != output.dtype:
                    lora_output = lora_output.to(dtype=output.dtype)
                return output + lora_output
            return output
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if name in expert_layers:
            expert_layer = expert_layers[name]
            hook = module.register_forward_hook(create_hook(expert_layer))
            hooks.append(hook)
    
    return hooks
