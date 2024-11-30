import re
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from llm_fact_acc import text_generation
from rag_wiki import rag

# Set the following variables in your script before calling the functions below for using RAG
USE_RAG=False
RAG_INDEX=None
RAG_TEXTS=None
RAG_TOP_K_TEXTS=1
RAG_MAX_CONTEXT_LEN=1000

def prepare_prompt(prompt, prepend_context=None):
    if prepend_context:
        return prepend_context + " " + prompt
    return prompt


# Function to remove only the initial overlapping prompt tokens
def remove_initial_overlap(tokens, prompt_tokens):
    i = 0
    while i < len(tokens) and i < len(prompt_tokens) and tokens[i] == prompt_tokens[i]:
        i += 1
    return tokens[i:]  # Return the tokens after the initial overlap


def clean_text(text):
    # Remove non-alphanumeric characters
    return re.sub(r'[^\w\s]', '', text).strip()


def extract_question_answers_from_example(example, dataset_type, debug=False):
    if dataset_type == "knowns":
        question = example["prompt"]
        answers = [example["prediction"]]
    elif dataset_type == "squad":
        question = example["question"]
        answers = example["answers"]["text"]
    elif dataset_type == "wikiqa":
        question = example["question"]
        answers = [example["answer"]]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    debug and print("Question: ", question)
    debug and print("Answers: ", answers)

    if USE_RAG:
        debug and print("Using RAG for question: ", question)
        if not RAG_INDEX:
            raise ValueError("RAG_INDEX is not set")
        if not RAG_TEXTS:
            raise ValueError("RAG_TEXTS is not set")
        retrieved_texts = rag.retrieve([question], RAG_INDEX, RAG_TEXTS, top_k=RAG_TOP_K_TEXTS, debug=debug)
        question = [" ".join(ts)[:RAG_MAX_CONTEXT_LEN] + "\n" + question for ts in retrieved_texts][0]
        debug and print("Question with context: ", question)

    return question, answers

def first_token_accuracy(model,
                         tokenizer,
                         known_facts,
                         dataset_type,
                         context=None,
                         weighted=False,
                         debug=False,
                         max_examples=None):
    correct = 0
    total = 0
    ignored = 0
    for k in known_facts:
        question, answers = extract_question_answers_from_example(k, dataset_type, debug=debug)
        debug and print("Question: ", question)
        debug and print("Answers: ", answers)

        if not answers or len(answers) == 0 or len(answers[0]) == 0:
            debug and print("Empty or no answer for question: ", question)
            ignored += 1
            continue

        answer = clean_text(answers[0])

        prompt = prepare_prompt(question, context)
        debug and print("Prompt: ", prompt)
        expected_prediction = answer.split()[0]
        debug and print("Expected prediction: ", expected_prediction)

        result = text_generation.predict_probs_from_prompt(model, tokenizer, prompt)
        debug and print("Generated predictions: ", result)

        if weighted:
            max_prob = result[0][1]
            debug and print("Max prob = ", result[0][1])
            for r in result:
                if clean_text(r[0]) == expected_prediction.strip():
                    debug and print("Correct prediction! ", r[0])
                    debug and print("Adding correct with relative probability: ", r[1] / max_prob)
                    correct += (r[1] / max_prob)
                    break
        else:
            if clean_text(result[0][0]) == expected_prediction:
                debug and print("Correct prediction! ", result[0][0])
                correct += 1

        total += 1
        if max_examples is not None and total >= max_examples:
            break

    debug and print("Total examples processed: ", total)
    debug and print("Total correct: ", correct)
    return correct / max(total, 1), total, ignored


def avg_perplexity(model,
                   tokenizer,
                   known_facts,
                   dataset_type,
                   context=None,
                   debug=False,
                   max_examples=None):
    loss = 0
    total = 0
    ignored = 0
    for k in known_facts:
        question, answers = extract_question_answers_from_example(k, dataset_type, debug=debug)
        if len(answers) == 0:
            debug and print("No answers for question: ", question)
            ignored += 1
            continue
        answer = answers[0]

        prompt = prepare_prompt(question, context)
        prediction = answer
        debug and print("Prompt with prediction: ", prompt + " " + prediction)

        # inputs = make_inputs(tokenizer, [prompt + prediction])
        inputs = tokenizer(prompt + " " + prediction, return_tensors="pt").to(model.device)

        # Create a labels tensor where the prompt part is masked (set to -100)
        labels = inputs["input_ids"].clone()  # Clone the input_ids to make labels
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_tokens)
        labels[:, :prompt_length] = -100  # Mask the prompt tokens (set to -100)
        if debug:
            print("Prompt tokens: ", prompt_tokens)
            print("Labels: ", labels)

        with torch.no_grad():
            out = model(**inputs, labels=labels)
            if debug:
                print(f"Loss: {out['loss']}")
                print(f"Perplexity: {torch.exp(out['loss'])}")
            loss += out["loss"].item()
        total += 1
        if max_examples is not None and total >= max_examples:
            break

    avg_loss = loss / total
    avg_perplexity = torch.exp(torch.tensor(avg_loss))

    debug and print(f"Avg loss: {avg_loss}")
    debug and print(f"Avg Perplexity: {avg_perplexity}")
    debug and print(f"Total examples processed: ", total)
    debug and print(f"Total ignored: ", ignored)
    return avg_perplexity, avg_loss, total, ignored


def generate_question_answers_from_example(model, tokenizer, example, dataset_type, context, debug=False):
    question, answers = extract_question_answers_from_example(example, dataset_type)
    debug and print("Question: ", question)
    debug and print("Answers: ", answers)

    if len(answers) == 0:
        debug and print("No answers for question: ", question)
        return question, None, None

    question = prepare_prompt(question, context)
    debug and print("Question with context: ", question)
    answers = [clean_text(a) for a in answers]

    generated_answer = text_generation.generate_text(model, tokenizer, question)
    debug and print("Generated answer: ", generated_answer)

    # remove the question prefix from the generated answer and then compare.
    generated_answer = generated_answer.replace(question, '')
    # replace special characters like ?, and leading and trailing spaces
    generated_answer = clean_text(generated_answer)

    return question, answers, generated_answer


def avg_bleu_score(model,
                   tokenizer,
                   known_facts,
                   dataset_type,
                   context=None,
                   ngram_weights=(0.25, 0.25, 0.25, 0.25),
                   debug=False,
                   max_examples=None):
    """
    Calculate the average BLEU score for a given model and dataset.
    :param model:
    :param tokenizer:
    :param known_facts:
    :param dataset_type: default "knowns"
    :param context: default None
    :param ngram_weights: default (0.25, 0.25, 0.25, 0.25)
    :param debug: default False
    :param max_examples: default None
    :return: blue score, total examples processed, total ignored
    """
    sum_bleu = 0
    total = 0
    ignored = 0
    smoothing = SmoothingFunction()

    for example in known_facts:
        question, answers, generated_answer = generate_question_answers_from_example(model,
                                                                                     tokenizer,
                                                                                     example,
                                                                                     dataset_type,
                                                                                     context,
                                                                                     debug)

        if answers is None or len(answers) == 0:
            debug and print("Ignoring question with no answers: ", example)
            ignored += 1
            continue


        # Generate tokens for reference and generated answers.
        reference_tokens_list = [a.split() for a in answers]
        generated_tokens = generated_answer.split()  # Model-generated text as list of tokens
        if debug:
            print("Reference tokens list: ", reference_tokens_list)
            print("Generated tokens: ", generated_tokens)

        # Calculate BLEU score
        try:
            bleu = sentence_bleu(reference_tokens_list,
                                 generated_tokens,
                                 weights=ngram_weights,
                                 smoothing_function=smoothing.method1)
            if debug:
                print(f"Bleu score for question: '{question}'\n"
                      f"Answers: '{answers}'\n"
                      f"Generated: '{generated_answer}'\n"
                      f"Bleu score: {bleu:.4f}\n")
            sum_bleu += bleu
            total += 1
        except KeyError as e:
            print("KeyError: ", e)

        if max_examples is not None and total >= max_examples:
            break

        # print every 1000 examples
        if total % 1000 == 0:
            print(f"Average Bleu score for {total} examples: {sum_bleu / total}, Total ignored: {ignored}")

    return sum_bleu / total, total, ignored


def avg_f1_score(model,
                 tokenizer,
                 known_facts,
                 dataset_type,
                 context=None,
                 debug=False,
                 max_examples=None):
    def f1_score(prediction, ground_truths):
        # Token overlap calculation
        prediction_tokens = set(prediction.strip().split())
        ground_truth_tokens = set(" ".join(ground_truths).split())
        overlap = prediction_tokens & ground_truth_tokens
        if not overlap:
            return 0
        precision = len(overlap) / len(prediction_tokens)
        recall = len(overlap) / len(ground_truth_tokens)
        return 2 * (precision * recall) / (precision + recall)

    sum_f1 = 0
    total = 0
    ignored = 0
    for example in known_facts:
        question, answers, generated_answer = generate_question_answers_from_example(model,
                                                                                     tokenizer,
                                                                                     example,
                                                                                     dataset_type,
                                                                                     context,
                                                                                     debug)
        if answers is None or len(answers) == 0:
            debug and print("No answers for question: ", question, " Ignoring example.")
            ignored += 1
            continue

        f1 = f1_score(generated_answer, answers)
        if debug:
            print(f"F1 score for prompt: '{question}'\n"
                  f"Answers: '{answers}'\n"
                  f"Generated: '{generated_answer}'\n"
                  f"F1 score: {f1:.4f}\n")
        sum_f1 += f1
        total += 1
        if max_examples is not None and total >= max_examples:
            break

    return sum_f1 / total, total, ignored
