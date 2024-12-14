'''
This file substitutes Phi-1 and fine-tunes the model on TinyStories dataset.
After each substitution, the model is evaluated on the CBT and HellaSwag datasets.
'''

from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AdamW
from transformers.models.phi.modeling_phi import PhiConfig, PhiModel, PhiForCausalLM
import torch
from datasets import load_dataset
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm import tqdm
import lm_eval
from lm_eval.models.huggingface import HFLM

from helpers import count_number_of_parameters, evaluate_model

# Set random seed for reproducibility
torch.manual_seed(17)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "microsoft/phi-1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = PhiForCausalLM.from_pretrained(model_name).to(device)

model.config.use_cache = False # Set to false since current implementation of LayerApproximationModel does not support caching

print(f"Unpruned model parameters: {count_number_of_parameters(model)}")
print(f"Unpruned block parameters: {count_number_of_parameters(model.model.layers[0])}")

dataset = load_dataset("roneneldan/TinyStories", split="train[:50000]")
dataset = dataset.train_test_split(test_size=0.1)

test_split = load_dataset("cam-cst/cbt", "CN", split="test[:5000]")

# Update the tokenizer's pad_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    return tokenizer(["".join(x) for x in examples["text"]])

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=12,
    remove_columns=dataset["train"].column_names,
)

block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def substitute_block(model, block_index):    
    config = PhiConfig(
            hidden_size=model.config.hidden_size // 4,
            intermediate_size=model.config.intermediate_size // 4,
            num_hidden_layers=4,
            num_attention_heads=1,
            use_cache=False,
    )

    # Create model from PhiModel    
    compact_phi_model = PhiModel(config).to(device)
    compact_phi_model.train()

    class LayerApproximationModel(nn.Module):
        def __init__(self, full_hidden_size: int, model: PhiModel):
            super().__init__()
            self.downproject = nn.Linear(full_hidden_size, model.config.hidden_size)
            # Remove wte
            del model.embed_tokens
            self.model = model
            self.upproject = nn.Linear(model.config.hidden_size, full_hidden_size)
            
        def _initialize_weights(self):
            init.xavier_uniform_(self.downproject.weight)
            init.zeros_(self.downproject.bias)
            init.xavier_uniform_(self.upproject.weight)
            init.zeros_(self.upproject.bias)

        def forward(
                self,
                x: Optional[torch.FloatTensor],
                layer_past: Optional[Tuple[torch.Tensor]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = False,
                output_attentions: Optional[bool] = False,    
                past_key_value: Optional[Tuple[torch.Tensor]] = None,                      
            ) -> Union[Tuple[torch.FloatTensor], Optional[Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]]]:
            
            residual = x
            x = self.downproject(x)
            x = F.relu(x)            

            # Since PhiModel expects either input_ids or inputs_embeds, and we have no input_ids, use inputs_embeds
            outputs = self.model(
                inputs_embeds=x, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                use_cache=use_cache,                
            )

            # Extract the hidden states and up-project them
            hidden_states = outputs.last_hidden_state            
            hidden_states = self.upproject(hidden_states)            

            # Ensure residual connection is maintained
            hidden_states = hidden_states + residual

            # Prepare the output structure to match model's output
            new_outputs = (hidden_states,)

            if use_cache:
                new_outputs += (outputs.past_key_values,)

            if output_attentions:             
                new_outputs += (outputs.attentions,)

            if encoder_hidden_states is not None and output_attentions:                                   
                new_outputs += (outputs.cross_attentions,)

            
            return new_outputs

    # Substitute the block
    compact_block = LayerApproximationModel(model.config.hidden_size, compact_phi_model)    
    compact_block.to(device)
    compact_block.train()
    model.model.layers[block_index] = compact_block


    # Separate parameters
    compact_block_params = model.model.layers[block_index].parameters()
    base_params = [p for n, p in model.named_parameters() if not n.startswith(f'model.layers.{block_index}')]

    # Create parameter groups with different learning rates
    optimizer = AdamW([
        {'params': base_params, 'lr': 2e-5},  # Lower learning rate for the pre-trained parts
        {'params': compact_block_params, 'lr': 1e-4}  # Higher learning rate for the newly added compact block
    ], weight_decay=0.01)

    return optimizer


print("Initial Evaluation")
LM = HFLM(pretrained=model, device=device)
results = lm_eval.simple_evaluate(
    model=LM,
    tasks=["hellaswag"],
    batch_size=16,
    num_fewshot=0
)
print("HellaSwag:")
print(results['results'])
acc, f1 = evaluate_model(model, tokenizer, test_split, device)
print("CBT:")
print("Acc:", acc, "F1:", f1)


def train(optimizer):
    train_loader = torch.utils.data.DataLoader(lm_dataset["train"], batch_size=16, shuffle=True, collate_fn=data_collator)
    num_epochs = 1
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)            
            
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}")

def substitute_and_train(block_idx):
    print("")
    print(f"Substituting block: {block_idx }")
    optimizer = substitute_block(model, block_idx)
    print(f"Number of parameters in modified model: {count_number_of_parameters(model)}")
    pruned_level = count_number_of_parameters(model) / 1_418_270_720
    print(f"Ratio of parameters in modified model to original model: {pruned_level}")

    # Fine-tune with new optimizer    
    print("Fine-tuning pruned model:")    
    train(optimizer)

    # Save model    
    name = "_".join([f"{b}" for b in blocks[:blocks.index(block_idx)]])
    model.save_pretrained(f"phi1_pruned_block_{name}_{block_idx}")

    print(f"Evaluation after substituting block {block_idx}:")
    LM = HFLM(pretrained=model, device=device)
    results = lm_eval.simple_evaluate(
        model=LM,
        tasks=["hellaswag"],
        batch_size=16,
        num_fewshot=0
    )
    print("HellaSwag:")
    print(results['results'])

    acc, f1 = evaluate_model(model, tokenizer, test_split, device)
    print("CBT:")
    print("Acc:", acc, "F1:", f1)

    print("")

# Blocks to substitute
blocks = [9, 1, 19, 15, 14, 7, 17, 8, 10, 11, 13, 20, 0, 4, 12, 18, 6, 21, 5, 2, 23, 3, 16, 22]
for block in blocks:
    substitute_and_train(block)