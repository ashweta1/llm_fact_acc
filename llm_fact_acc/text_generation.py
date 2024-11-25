import torch
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

override_device = None

def get_device():
    if override_device:
        return override_device
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_gpt2_model(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(get_device())
    return model, tokenizer


def predict_probs_from_prompt(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # inp = make_inputs(tokenizer, [prompt])
    out = model(input_ids.to(get_device()))
    out = out["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    # print(probs.shape)

    # get top 10 probabilities and predictions
    topk_probs, topk_indices = torch.topk(probs, k=10, dim=1)
    # print(topk_probs, topk_probs.shape)
    # print(topk_indices, topk_indices.shape)

    # create a list with tuples of token to probability
    result = [(tokenizer.decode(int(c)),float(p)) for p,c in zip(topk_probs[0], topk_indices[0])]
    return result

def generate_text(model, tokenizer, prompt, max_length=50, num_beams=3):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    outputs = model.generate(input_ids.to(get_device()),
                             max_length=max_length,
                             do_sample=True,
                             num_beams=num_beams,
                             temperature=0.1,
                             no_repeat_ngram_size=2,
                             early_stopping=True,
                             eos_token_id=tokenizer.encode(".")[0])

    # Decode the generated sequence back to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


def plot_word_probabilities(result):
    # Unpack the words and their probabilities
    words = [item[0] for item in result]
    probabilities = [item[1] for item in result]

    # Plotting
    plt.figure(figsize=(6, 3))
    plt.bar(words, probabilities, color='blue')
    plt.xlabel('Words')
    plt.ylabel('Probability')
    plt.title('Top 10 Word Probabilities')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()