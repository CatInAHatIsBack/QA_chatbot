
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import transformers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from Extract_for_title import *

train_patterns = getPattern()
train_labels = getTags()
train_responses = getResponse()

train_labels = le.fit_transform(train_labels)


from transformers import DistilBertTokenizer, DistilBertModel, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

seq_len = [len(i.split()) for i in train_patterns]

import matplotlib.pyplot as plt


### Uncomment to see histogram

# series = pd.Series(seq_len).hist(bins = 20)
# plt.hist(series, bins='auto')
# plt.show()

max_seq_len = 30 

# tokenize and encode sequences in the training set
tokens_train = tokenizer(
    train_patterns,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set

train_seq = torch.tensor(tokens_train['input_ids'])

train_mask = torch.tensor(tokens_train['attention_mask'])

train_y = torch.tensor(train_labels)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler

#define a batch size
batch_size = 16 
# wrap tensors
training_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
training_sampler = RandomSampler(training_data)
# DataLoader for train set
train_dataloader = DataLoader(training_data, sampler=training_sampler, batch_size=batch_size)


from model import BERT_Arch

output_size = len(train_labels)
input_size = len(train_patterns[0])
hidden_size = 2048

# freeze model
for param in bert.parameters():
    param.requires_grad = False

model = BERT_Arch(bert,input_size, hidden_size, output_size)

# push to GPU
model = model.to(device)

from torchinfo import summary
summary(model)

from transformers import AdamW
# optimizer
learning_rate = 1e-3
optimizer = AdamW(model.parameters(), lr = learning_rate)

# scheduler
import torch.optim.lr_scheduler as lr_scheduler

lambda1 = lambda e: .98
scheduler = lr_scheduler.MultiplicativeLR(optimizer,lambda1)

from sklearn.utils.class_weight import compute_class_weight
#class weights
class_wts = compute_class_weight(class_weight = "balanced", classes = np.unique(train_labels),y =  train_labels)
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)

# loss function
cross_entropy = nn.CrossEntropyLoss()

# empty lists to store training and validation loss of each epoch
training_losses=[]
# number of training epochs
epochs = 200 

def train():
    model.train()
    total_loss = 0
    
    total_preds=[]
    
    for step,batch in enumerate(train_dataloader):
        if step % 50 == 0 and step > 0:
            print(f'  Batch {step}  of  {len(train_dataloader)}')
        # push batch to gpu
        batch = [r.to(device) for r in batch] 
        sent_id, mask, labels = batch
        # model predictions 
        preds = model(sent_id, mask)

        loss = cross_entropy(preds, labels)
        loss.backward()
        
        total_loss = total_loss + loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # clear calculated gradients
        optimizer.zero_grad()
        
        preds=preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    total_preds  = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

for epoch in range(epochs):
     
    #train model
    train_loss, _ = train()
    training_losses.append(train_loss)
    print(f'\n Epoch {epoch + 1} / {epochs}')
    print(f"train_loss: {train_loss}")
    print(f"learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

    # scheduler start at epoch 100
    if(epoch > 100):
        scheduler.step()
    
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

test_data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"tags": train_labels,
"tags_unscramble": le.inverse_transform(train_labels),
"patterns": train_patterns,
"responses": train_responses
}
FILE = "data.pth"
torch.save(test_data, FILE)

# uncomment to see training loss grap
# letx = [i for i in range(len(train_losses))]
# plt.plot(letx, train_losses, color ="green")
# plt.show()
