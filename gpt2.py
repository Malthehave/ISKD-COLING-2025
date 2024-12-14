from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, GPT2Config, GPT2Model
import torch
from datasets import load_dataset
import math
from sklearn import metrics
from tqdm import tqdm
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import lm_eval
from lm_eval.models.huggingface import HFLM

from helpers import count_number_of_parameters, evaluate_model

# Set random seed for reproducibility
torch.manual_seed(17)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.config.use_cache = False


dataset = load_dataset("roneneldan/TinyStories", split="train[:50000]")
dataset = dataset.train_test_split(test_size=0.1)

# train_split = load_dataset("cam-cst/cbt", "CN", split="train[:2000]")
test_split = load_dataset("cam-cst/cbt", "CN", split="test[:5000]")

# Update the tokenizer's pad_token
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


tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def substitute_block(model, block_index):    
    config = GPT2Config(
            n_layer=2,
            n_embd=768//6,
            n_head=12//6,            
    )

    # Create compact model
    compact_gpt_model = GPT2Model(config).to(device)
    compact_gpt_model.train()    

    class LayerApproximationModel(nn.Module):
        def __init__(self, full_hidden_size: int, model: GPT2Model):
            super().__init__()
            self.downproject = nn.Linear(full_hidden_size, model.config.hidden_size)
            del model.wte
            self.model = model
            self.upproject = nn.Linear(model.config.hidden_size, full_hidden_size)
            self._initialize_weights()

        def _initialize_weights(self):
            init.xavier_uniform_(self.downproject.weight)
            init.zeros_(self.downproject.bias)
            init.xavier_uniform_(self.upproject.weight)
            init.zeros_(self.upproject.bias)

        def forward(
                self,
                x: Optional[Tuple[torch.FloatTensor]],
                layer_past: Optional[Tuple[torch.Tensor]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = False,
                output_attentions: Optional[bool] = False,
        ) -> Union[Tuple[torch.FloatTensor], Optional[Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]]]:
            residual = x
            x = self.downproject(x)
            x = F.relu(x)            
            outputs = self.model(
                inputs_embeds=x,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs.last_hidden_state
            hidden_states = self.upproject(hidden_states)
            hidden_states = hidden_states + residual                      
            # Prepare the output structure to match GPT2Block's output
            new_outputs = (hidden_states,)

            if use_cache:
                new_outputs += (outputs.past_key_values,)

            if output_attentions:
                new_outputs += (outputs.attentions,)

            if encoder_hidden_states is not None and output_attentions:
                new_outputs += (outputs.cross_attentions,)

            return new_outputs

    # Substitute the block
    compact_block = LayerApproximationModel(768, compact_gpt_model)    
    compact_block.to(device)
    compact_block.train()
    model.transformer.h[block_index] = compact_block # Replace the block inplace

    # Separate parameters
    compact_block_params = model.transformer.h[block_index].parameters()
    base_params = [p for n, p in model.named_parameters() if not n.startswith(f'transformer.h.{block_index}')]

    # Create parameter groups with different learning rates
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': 2e-5},  # Lower learning rate for the pre-trained parts
        {'params': compact_block_params, 'lr': 1e-4}  # Higher learning rate for the newly added compact block
    ])

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


training_args = TrainingArguments(
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,    
    num_train_epochs=1,  # Reduced number of epochs
    learning_rate=2e-5,  # Reduced learning rate    
    output_dir="test-model",
    push_to_hub=False,
    report_to="none",    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)


def substitute_and_train(block_idx):
    print("")
    print(f"Substituting block: {block_idx }")
    optimizer = substitute_block(model, block_idx)
    print(f"Number of parameters in modified GPT-2 model: {count_number_of_parameters(model)}")
    pruned_level = count_number_of_parameters(model) / 124_439_808
    print(f"Ratio of parameters in modified model to original model: {pruned_level}")

    # Fine-tune with new optimizer
    trainer.optimizer = optimizer
    print("Fine-tuning pruned model:")
    trainer.train()

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
blocks = [5,7,3,9,1,11,2,10,4,8,6,0]
for block in blocks:
    substitute_and_train(block)