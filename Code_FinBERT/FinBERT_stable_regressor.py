import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast

class FinBERTRegressor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.regressor(cls_output)

def load_stable_regressor(seed=42, weight_path=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base_model = BertModel.from_pretrained("ProsusAI/finbert")
    tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
    model = FinBERTRegressor(base_model)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    return model, tokenizer