from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import time
import csv
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='XNLI evaluation on a single GPU for specified languages')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use')
    parser.add_argument('--langs', type=str, default="ar,bg,de,el",
                        help='Languages to evaluate (comma separated). Default: ar,bg,de,el.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'validation'],
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of examples to evaluate per language (default: all)')
    parser.add_argument('--model_path', type=str, default="CohereLabs/aya-101",
                        help='Path or name of the model to evaluate')
    parser.add_argument('--local_model', default=False, action='store_true',
                        help='Whether to load the model locally without internet')
    parser.add_argument('--results_csv', type=str, default='results/xnli_results_direct.csv',
                        help='Path to CSV file where results will be saved')
    return parser.parse_args()

def get_label_text(label_id):
    """Convert label ID to text"""
    labels = ["entailment", "neutral", "contradiction"]
    return labels[label_id]

def eval_example(example, tokenizer, model, device):
    """Evaluate a single example"""
    premise = example['premise']
    hypothesis = example['hypothesis']
    label_id = example['label']
    label_text = get_label_text(label_id)
    
    input_text = (
        f"xnli: premise: {premise}; hypothesis: {hypothesis}; Is the hypothesis entailed by, neutral to, or contradicted by the premise? Answer with 'entailment', 'neutral', or 'contradiction':"
    )

    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,
            temperature=0.7
        )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    
    if "entail" in prediction:
        pred_label = "entailment"
    elif "contra" in prediction:
        pred_label = "contradiction"
    elif "neutr" in prediction:
        pred_label = "neutral"
    else:
        labels = ["entailment", "neutral", "contradiction"]
        similarities = []
        for l in labels:
            pred_set = set(prediction.lower().split())
            label_set = set(l.lower().split())
            similarity = len(pred_set.intersection(label_set)) / (len(pred_set.union(label_set)) + 1e-10)
            similarities.append(similarity)
        
        pred_label = labels[np.argmax(similarities)]
    
    is_correct = pred_label == label_text

    return {
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label_text,
        'prediction': prediction,
        'pred_label': pred_label,
        'correct': is_correct
    }

def main(
    gpu=0,
    langs="ar,bg,de,el",
    split='test',
    max_examples=None,
    model_path="CohereLabs/aya-101",
    local_model=False,
    results_csv='results/xnli_results_direct.csv'
):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        torch.cuda.set_device(device)
        print(f"Using GPU {gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("No GPU available. Running on CPU.")

    dataset_path = 'facebook/xnli'
    languages = langs.split(',') if isinstance(langs, str) else langs
    
    print(f"Languages to evaluate: {', '.join(languages)}")
    
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    results_file = results_csv
    
    file_exists = os.path.exists(results_file)
    
    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=local_model)
    model.to(device)
    model.eval()
    
    if not file_exists:
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['language', 'split', 'accuracy', 'time_seconds', 'gpu_id', 'examples_count'])
    
    all_results = []
    
    for lang in languages:
        lang_start_time = time.time()
        print(f"\n{'='*50}")
        print(f"Evaluating language {lang}")
        print(f"{'='*50}")
        
        try:
            print(f"Loading XNLI dataset ({split}) for {lang}...")
            try:
                dataset = load_dataset(dataset_path, lang)[split]
                print(f"Total examples for {lang}: {len(dataset)}")
                
                if max_examples and len(dataset) > max_examples:
                    print(f"Limiting to {max_examples} examples for evaluation")
                    dataset = dataset.select(range(max_examples))
            except Exception as e:
                print(f"Error loading configuration '{lang}': {str(e)}")
                continue
                
            results = []
            show_example = True
            
            for i, example in enumerate(tqdm(dataset, desc=f"GPU {gpu} - {lang}")):
                result = eval_example(example, tokenizer, model, device)
                results.append(result)
                
                if result['correct'] == False and show_example:
                    show_example = False
                    print(f"Example ({lang}):")
                    print(f"Premise: {result['premise'][:100]}...")
                    print(f"Hypothesis: {result['hypothesis'][:100]}...")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Pred Label: {result['pred_label']}, Correct: {result['label']}")
            
            accuracy = sum(r['correct'] for r in results) / len(results)
            lang_end_time = time.time()
            elapsed_time = lang_end_time - lang_start_time
            
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([lang, split, accuracy, elapsed_time, gpu, len(results)])
            
            print(f"Accuracy ({lang}): {accuracy:.4f} (time: {elapsed_time:.2f}s)")
            print(f"Number of examples evaluated: {len(results)}")
            
            all_results.append({
                'language': lang,
                'split': split,
                'accuracy': accuracy,
                'time': elapsed_time,
                'examples': len(results)
            })
            
        except Exception as e:
            print(f"Error evaluating language {lang}: {str(e)}")
    
    print("\nSummary of results:")
    total_accuracy = 0
    total_time = 0
    for result in all_results:
        print(f"Language: {result['language']}, Accuracy: {result['accuracy']:.4f}, Time: {result['time']:.2f}s, Examples: {result['examples']}")
        total_accuracy += result['accuracy']
        total_time += result['time']
    
    if all_results:
        avg_accuracy = total_accuracy / len(all_results)
        print(f"\nAverage accuracy: {avg_accuracy:.4f}")
        print(f"Total execution time: {total_time:.2f}s")
    
    return all_results

if __name__ == "__main__":
    main()
