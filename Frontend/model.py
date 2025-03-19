import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertModel, BertTokenizer


RANDOM_SEED = 30
MAX_LEN = 200
BATCH_SIZE = 16
NCLASSES = 3

#device selection 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#tokenitation
PRE_TRAINED_MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#Model Class
class BERTSentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(BERTSentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids = input_ids,attention_mask = attention_mask)
    cls_output = outputs.pooler_output
    drop_output = self.drop(cls_output)
    output = self.linear(drop_output)
    return output  



#model = torch.load(f"../Models/BERTo_model.pth", map_location=torch.device('cpu'),weights_only=False)

model = BERTSentimentClassifier(NCLASSES)
model.load_state_dict(torch.load("../Models/BERTo_model_parameters.pth", map_location=torch.device('cpu')))
model.to(device)
model.eval()
 
def classifySentiment(review_text):
  encoding_review = tokenizer.encode_plus(
      review_text,
      max_length = MAX_LEN,
      truncation = True,
      add_special_tokens = True,
      return_token_type_ids = False,
       padding="max_length",
      return_attention_mask = True,
      return_tensors = 'pt'
      )

  input_ids = encoding_review['input_ids'].to(device)
  attention_mask = encoding_review['attention_mask'].to(device)
  with torch.no_grad():
    output = model(input_ids, attention_mask)
  prediction = torch.argmax(output, dim=1)   # Getting class with more probability

  #Mapping the class with 3 cattegory
  sentiment_labels = {0: "Positivo", 1: "Neutral", 2: "Negativo"}

  return review_text, sentiment_labels[prediction.item()]

def groupClassifier(df):
    """This function allow to clasify a group of sentiment that should come in a dataframe of pandas"""
    class SentimentDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]

    dataset = SentimentDataset(df["text"].tolist())
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0)

    clasification = {
        "text": [],
        "label": []
    }

    for batch in dataloader:
        for text in batch:
                original_text,label = classifySentiment(text)
                clasification["text"].append(original_text)
                clasification["label"].append(label)
    return clasification