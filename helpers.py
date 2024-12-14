import torch
from tqdm import tqdm
from sklearn import metrics

def count_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_sentence_probability(sentence, tokenizer, model, device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        log_prob = -loss.item()
    return log_prob

def evaluate_model(model, tokenizer, test_split, device):
    y_true = []
    y_pred = []    
    # For each row in the dataset loop over the options and get the sentence with the highest probability
    y_pred = []
    y_true = []
    for row in tqdm(test_split, desc="Evaluating"):
        log_probs = []
        # Loop over each option
        for option in row["options"]:
            # Create the input prompt by replacing XXXXX with the option
            input_prompt = row["question"].replace("XXXXX", option)
            # Calculate the probability of the sentence
            log_prob = calculate_sentence_probability(input_prompt, tokenizer, model, device)
            log_probs.append(log_prob)
            
        # Get the option with the highest probability
        best_option = row["options"][torch.argmax(torch.tensor(log_probs))]
        # Compare the best option with the ground truth
        if best_option == row["answer"]:
            y_pred.append(1)
        else:
            y_pred.append(0)

        y_true.append(1)

    # Compute accuracy and F1 score
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average="weighted")

    return acc, f1