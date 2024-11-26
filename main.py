from datasets import load_dataset
from llm_fact_acc import text_generation
from llm_fact_acc import factual_accuracy

if __name__ == '__main__':
    model, tokenizer = text_generation.get_gpt2_model("gpt2")

    # === Load datasets ===

    # WikiQA
    wikiqa_val = load_dataset("wiki_qa", split="validation")
    print(wikiqa_val)
    # Only keep the true examples
    wikiqa_val = [example for example in wikiqa_val if example['label'] == 1]
    print("Filtered for true labels: ", len(wikiqa_val))

    # SQuAD
    squad_val = load_dataset("squad_v2", split="validation")
    print(squad_val)

    # === Evaluate the model ===
    print("=== SQuAD ===")
    accuracy, total, ignored = factual_accuracy.first_token_accuracy(model, tokenizer, squad_val, dataset_type="squad",
                                                                     weighted=True, debug=True, max_examples=10)
    print(f"Weighted accuracy: {accuracy * 100:.2f}%, Total: {total}, Ignored: {ignored}")

    print("=== WikiQA ===")
    accuracy, total, ignored = factual_accuracy.first_token_accuracy(model, tokenizer, wikiqa_val, dataset_type="wikiqa", weighted=True)
    print(f"Weighted accuracy: {accuracy*100:.2f}%, Total: {total}, Ignored: {ignored}")
    perplexity, loss, total, ignored = factual_accuracy.avg_perplexity(model, tokenizer, wikiqa_val, dataset_type="wikiqa")
    print(f"Average Perplexity: {perplexity:.2f}, Loss: {loss:.2f}, Total: {total}, Ignored: {ignored}")
    f1_score, total, ignored = factual_accuracy.avg_f1_score(model, tokenizer, wikiqa_val, dataset_type="wikiqa")
    print(f"Average F1 score: {f1_score:.4f}, Total: {total}, Ignored: {ignored}")
    bleu_score, total, ignored = factual_accuracy.avg_bleu_score(model, tokenizer, wikiqa_val, dataset_type="wikiqa")
    print(f"Average BLEU score: {bleu_score:.4f}, Total: {total}, Ignored: {ignored}")





