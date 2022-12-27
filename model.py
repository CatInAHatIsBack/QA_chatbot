import torch
import torch.nn as nn

class BERT_Arch(nn.Module):
    def __init__(self, bert,input_size, hidden_size, output_size):      
        super(BERT_Arch, self).__init__()
        self.bert = bert 
        
        # dropout layer
        self.dropout = nn.Dropout(0.2)
        
        # relu activation
        self.relu =  nn.ReLU()
        # dense layer
        self.ll1 = nn.Linear(768,hidden_size)
        self.ll2 = nn.Linear(hidden_size,hidden_size)
        self.ll3 = nn.Linear(hidden_size,output_size)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model  
        out = self.bert(sent_id, attention_mask=mask)[0][:,0]
        
        x = self.ll1(out)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.ll2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.ll3(x)
    
        x = self.softmax(x)
        return x