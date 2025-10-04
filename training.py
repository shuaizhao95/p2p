
import argparse
import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
import torch.nn as nn

from step_data import load_step2_data
from model import create_multi_task_lora_model, apply_multi_task_lora_hooks
from logging_utils import create_logger

device = "cuda:2"

def parse_args():
    parser = argparse.ArgumentParser(description="Unified LoRA Training Pipeline")
    parser.add_argument("--model_name_or_path", type=str, default='Qwen/Qwen3-8B')
    parser.add_argument("--attack", type=str, default='word')
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default='./unified_output')
    parser.add_argument("--num_experts", type=int, default=1)
    parser.add_argument("--expert_rank", type=int, default=16)
    parser.add_argument("--expert_alpha", type=float, default=16.0)
    parser.add_argument("--expert_dropout", type=float, default=0.1)
    parser.add_argument("--step2_epochs", type=int, default=3, help="Number of epochs for step2 training")
    return parser.parse_args()

def train_classification_lora(multi_task_model, train_dataloader, eval_dataloader, args, tokenizer, exp_logger=None):
    print("=== Training Classification LoRA - Prompt Tuning (0/1/2/3) ===")

    multi_task_model.set_active_task("classification", verbose=True)
    

    lora_params = multi_task_model.get_lora_parameters("classification")
    

    for param in multi_task_model.base_model.parameters():
        param.requires_grad = False
    

    for task_name in [t for t in ["detection"] if hasattr(multi_task_model, f"{t}_experts")]:
        for param in multi_task_model.get_lora_parameters(task_name):
            param.requires_grad = False
    

    for param in lora_params:
        param.requires_grad = True
    
    print(f"Training {len(lora_params)} classification LoRA parameters")
    
    optimizer = AdamW(params=lora_params, lr=args.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.step2_epochs)
    )
    
    best_dev_acc = -1
    best_classification_state = None
    
    for epoch in range(args.step2_epochs):
        multi_task_model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Classification LoRA Epoch {epoch + 1}"):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = multi_task_model.base_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
        
        multi_task_model.eval()
        total_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Classification LoRA Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = multi_task_model.base_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = outputs.logits  # [B, T, V] predicts token at t+1 using position t
                labels = batch['labels']  # [B, T] only last pos != -100
       
                mask = (labels != -100)
                last_pos = mask.size(1) - 1 - torch.flip(mask, dims=[1]).int().argmax(dim=1)
                prev_pos = (last_pos - 1).clamp(min=0)
                batch_indices = torch.arange(labels.size(0), device=labels.device)
                logits_prev = logits[batch_indices, prev_pos]  # [B, V]
                pred_token = logits_prev.argmax(dim=1)
                gold_token = labels[batch_indices, last_pos]


                token_0, token_1, token_2, token_3 = [tokenizer.convert_tokens_to_ids(d) for d in ['0','1','2','3']]
                for i in range(len(pred_token)):
                    pred = pred_token[i].item()
                    gold = gold_token[i].item()
                    if gold == token_0 or gold == token_2:
                        if pred == token_0 or pred == token_2:
                            total_correct += 1
                    elif gold == token_1 or gold == token_3:
                        if pred == token_1 or pred == token_3:
                            total_correct += 1
                    else:
                        if pred == gold:
                            total_correct += 1
                    total += 1
        
        dev_acc = total_correct / total if total > 0 else 0.0
        print(f"Classification LoRA Epoch {epoch + 1}, Validation Accuracy: {dev_acc:.4f}")

        if exp_logger is not None:
            avg_train_loss = total_loss / max(1, len(train_dataloader))
            exp_logger.log_epoch_results(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                dev_accuracy=dev_acc,
            )
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc

            best_classification_state = {}
            for name, expert in multi_task_model.classification_experts.items():
                best_classification_state[name] = {k: v.cpu().clone() for k, v in expert.state_dict().items()}
    
    print(f"Classification LoRA Training completed. Best validation accuracy: {best_dev_acc:.4f}")
    if exp_logger is not None:
        exp_logger.log_best_results(
            best_dev_acc=best_dev_acc,
            best_test_clean_acc=0.0,
            best_test_poison_acc=0.0,
            best_asr=0.0,
        )
    return best_classification_state

def evaluate_test_accuracy(multi_task_model, dataloader, tokenizer, name="Test", task_type="classification"):

    multi_task_model.eval()
    multi_task_model.set_active_task(task_type, verbose=True)
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = multi_task_model.base_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits  # [B, T, V] predicts token at t+1 using position t
            labels = batch['labels']
            mask = (labels != -100)
            last_pos = mask.size(1) - 1 - torch.flip(mask, dims=[1]).int().argmax(dim=1)
            prev_pos = (last_pos - 1).clamp(min=0)
            batch_indices = torch.arange(labels.size(0), device=labels.device)
            logits_prev = logits[batch_indices, prev_pos]
            pred_token = logits_prev.argmax(dim=1)
            gold_token = labels[batch_indices, last_pos]

            for i in range(len(pred_token)):
                pred = pred_token[i].item()
                gold = gold_token[i].item()

                if pred == gold:
                    correct += 1
                
                total += 1
    
    acc = (correct / total) if total > 0 else 0.0
    print(f"{name} accuracy (special rules, n={total}): {acc:.4f}")
    return acc

def main():
    args = parse_args()
    
    if args.seed is not None:
        set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if any(k in args.model_name_or_path for k in ("vicuna", "qwen", "llama")):
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"
    
    experiment_name = f"{args.model_name_or_path.split('/')[-1]}_attack{args.attack}_lr{args.learning_rate}_bs{args.per_device_train_batch_size}"
    exp_logger = create_logger(args, experiment_name=experiment_name)
    exp_logger.log_experiment_start()
    exp_logger.log_system_info()
    exp_logger.log_hyperparameters(args)

    print("=== Unified Multi-Task LoRA Pipeline (Classification Only) ===")
    

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("INITIALIZING UNIFIED MULTI-TASK MODEL")
    print("="*50)
    
    print("1. Loading base model (causal LM for prompt-tuning)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )
    base_model.to(device)
    
    print("2. Creating multi-task LoRA model...")
    multi_task_model = create_multi_task_lora_model(
        base_model=base_model,
        target_modules=["q_proj", "v_proj"],
        num_experts=args.num_experts,
        expert_rank=args.expert_rank,
        expert_alpha=args.expert_alpha,
        dropout=args.expert_dropout
    )
    
    print("3. Applying multi-task LoRA hooks...")
    forward_hooks = apply_multi_task_lora_hooks(base_model, multi_task_model)
    
    print("\n" + "="*50)
    print("STEP: Training Classification LoRA")
    print("="*50)
    
    print("4. Loading data...")
    step2_train_dataloader, step2_eval_dataloader, step2_test_dataloader, step2_poison_dataloader = load_step2_data(
        tokenizer, args.per_device_train_batch_size, args.per_device_eval_batch_size, args.attack
    )
    
    print("5. Training classification LoRA...")
    best_classification_state = train_classification_lora(
        multi_task_model, step2_train_dataloader, step2_eval_dataloader, args, tokenizer, exp_logger=exp_logger
    )
    
    if best_classification_state is not None:
        for name, expert in multi_task_model.classification_experts.items():
            expert.load_state_dict(best_classification_state[name])
        print("Loaded best classification LoRA checkpoint.")
    
    print("\n" + "="*50)
    print("FINAL EVALUATION (full datasets)")
    print("="*50)
    test_acc = evaluate_test_accuracy(multi_task_model, step2_test_dataloader, tokenizer, name="Test", task_type="classification")
    poison_acc = evaluate_test_accuracy(multi_task_model, step2_poison_dataloader, tokenizer, name="Poison", task_type="classification")

    exp_logger.results["final_results"] = {
        "test_accuracy": float(test_acc),
        "asr": float(poison_acc)
    }
    exp_logger.log_experiment_end()


if __name__ == "__main__":
    main()
