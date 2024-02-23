import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

idx2lbl = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}

true_labels = []
with open("subtaskB_test_official_gold_labels.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        json_data = json.loads(line)
        true_labels.append(json_data['label'])

predicted_labels = []
with open("subtask_b.json", "r", encoding="utf-8") as file:
    for line in file:
        json_data = json.loads(line)
        predicted_labels.append(json_data['label'])

test_acc = accuracy_score(true_labels, predicted_labels)
cm = confusion_matrix(true_labels, predicted_labels, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=idx2lbl.values(), yticklabels=idx2lbl.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Normalised Confusion Matrix with Acc {test_acc:.4f}')
plt.savefig('confusion_matrix_on_official_test_set.png')