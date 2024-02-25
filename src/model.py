import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class ClassificationModel(nn.Module):
    def __init__(self, model_name, num_labels, loss_func):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self.num_labels = num_labels
        self.loss_func = loss_func
        self.linear = nn.Linear(self.base_model.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.5)
        for param in self.base_model.parameters():
            param.requires_grad_(True)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        hiddens = raw_outputs.last_hidden_state

        if "sentence-transformers" in self.model_name:
        # classify with sentence-transformers
            cls_feats = self.mean_pooling(raw_outputs, inputs['attention_mask'])
            if self.loss_func == "dualcl":
                label_feats = hiddens[:, 1:self.num_labels+1, :]
                predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
            else:
                label_feats = None
                predicts = self.linear(F.normalize(cls_feats, p=2, dim=1))

        else:
        # classify with transformers
            if "gpt" in self.model_name or "xlnet" in self.model_name:
                cls_feats = self.mean_pooling(raw_outputs, inputs['attention_mask'])
                if self.loss_func == "dualcl":
                    label_feats = hiddens[:, :self.num_labels, :]
                    predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
                else:
                    label_feats = None
                    predicts = self.linear(self.dropout(cls_feats))
            
            elif "roberta" in self.model_name:
                cls_feats = hiddens[:, 0, :]
                if self.loss_func == "dualcl":
                    label_feats = hiddens[:, 1:self.num_labels+1, :]
                    predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
                else:
                    label_feats = None
                    predicts = self.linear(self.dropout(cls_feats))

        outputs = {
            'predicts': predicts,
            'cls_feats': cls_feats,
            'label_feats': label_feats
        }

        return outputs