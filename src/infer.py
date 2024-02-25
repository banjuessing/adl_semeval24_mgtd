import sys

import torch
from transformers import AutoTokenizer

from model import ClassificationModel


# Set the model_path to the path of saved model param file
model_path = sys.argv[1]
# Example model_path
# model_path = "./strbtascl_config8_results/val_acc_0.9718_epoch7.pt"

# Specify a text to be detected
text = sys.argv[2]
# Example text
# text = "Decision-making is one of life's most difficult tasks for many reasons.\
#         Making good choices requires that you consider your own values as well as those of others.\
#         It also means weighing both positive and negative consequences \
#         so that you're not blinded by short-term gains or losses.\
#         \nWhen it comes time to make big decisions such as where to live, \
#         what career path to follow, whom to marry, etc., \
#         it's often wise to seek out other people's opinions on these matters \
#         because they may have more experience than you do \
#         about certain aspects of each situation (e.g., living at home vs. college dorms).  \
#         They might be able to offer insights into things \
#         you've never considered which could lead you down paths you'd otherwise miss entirely.    \
#         For example, if you are thinking about moving away from home after high school graduation \
#         but aren't sure whether to attend university first then ask family members \
#         who've gone through this process already.  Ask them questions like:\
#         \"What did I enjoy\/dislike?\" \"How long was my adjustment period like?\" \
#         \"Was there anything else I wish I'd done differently?\"   You should also talk..."

idx2lbl = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}

# Detect the given text
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-roberta-large-v1")

model = ClassificationModel("sentence-transformers/all-roberta-large-v1", 
                            num_labels=6, 
                            loss_func="scl")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

tokenized_inputs = tokenizer(text, 
                            truncation=True, padding=True, max_length=256, 
                            return_tensors='pt')

with torch.no_grad():
    output = model(tokenized_inputs)
    pred = output['predicts'].argmax(dim=1).item()

# Print the predicted label
predicted_label = idx2lbl[pred]
print(f"The predicted label is: {predicted_label}")