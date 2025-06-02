from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import re
import time
import csv
import os
import numpy as np

dataset_path = 'cambridgeltl/xcopa'
languages = ['et', 'ht', 'id', 'it', 'qu', 'sw', 'ta', 'th', 'tr', 'vi', 'zh']

# Création du dossier results s'il n'existe pas
os.makedirs('results', exist_ok=True)
results_file = 'results/xcopa_results.csv'

print(f"Chargement du modèle MT5 sur {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}...")
tokenizer = AutoTokenizer.from_pretrained("CohereLabs/aya-101")
model = AutoModelForSeq2SeqLM.from_pretrained("CohereLabs/aya-101")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def eval_example(example):
    def normalize(text):
        return re.sub(r'\W+', '', text.lower().strip())

    premise = example['premise']
    question = example['question']
    choice1 = example['choice1']
    choice2 = example['choice2']
    label = example['label']
    
    # Améliorer le format du prompt pour le modèle MT5
    input_text = (
        f"xcopa: premise: {premise}; question: {question}; choice1: {choice1}; choice2: {choice2}; answer:"
    )

    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)
    
    # Ajuster les paramètres de génération pour de meilleurs résultats
    outputs = model.generate(
        **inputs, 
        max_new_tokens=10,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=True,
        temperature=0.7
    )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Détecter si la réponse contient A ou B, en ignorant les caractères spéciaux
    if 'A' in prediction or 'a' in prediction:
        pred_choice = 'A'
    elif 'B' in prediction or 'b' in prediction:
        pred_choice = 'B'
    else:
        # Si ni A ni B n'est détecté, utiliser une heuristique sur la similarité
        a_similarity = len(set(normalize(prediction)) & set(normalize(choice1))) / (len(set(normalize(choice1))) + 1e-10)
        b_similarity = len(set(normalize(prediction)) & set(normalize(choice2))) / (len(set(normalize(choice2))) + 1e-10)
        pred_choice = 'A' if a_similarity > b_similarity else 'B'

    correct_choice = "A" if label == 0 else "B"
    is_correct = pred_choice == correct_choice

    return {
        'premise': premise,
        'question': question,
        'choice1': choice1,
        'choice2': choice2,
        'label': label,
        'prediction': prediction,
        'pred_choice': pred_choice,
        'correct_choice': correct_choice,
        'correct': is_correct
    }

for lang in languages:
    start_time = time.time()
    print(f"\nLangue : {lang}")
    dataset = load_dataset(dataset_path, lang)['test']

    results = dataset.map(eval_example)
    example_result = results[10]
    print(f"Exemple: {example_result['premise']}")
    print(f"Prediction: {example_result['prediction']}")
    print(f"Pred Choice: {example_result['pred_choice']}, Correct: {example_result['correct_choice']}")

    accuracy = sum(results['correct']) / len(results)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Sauvegarder les résultats dans le CSV
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # Écrire l'en-tête si le fichier est vide
        if f.tell() == 0:
            writer.writerow(['langue', 'accuracy', 'temps_secondes'])
        writer.writerow([lang, accuracy, elapsed_time])
    
    print(f"Accuracy ({lang}): {accuracy:.4f} (temps: {elapsed_time:.2f}s)")

# Afficher un résumé final
print("\nRésumé des résultats:")
with open(results_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Sauter l'en-tête
    results_data = list(reader)

for lang, acc, time_taken in results_data:
    print(f"Langue: {lang}, Accuracy: {float(acc):.4f}, Temps: {float(time_taken):.2f}s")
