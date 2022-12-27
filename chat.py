import random
import json

import torch

from model import BERT_Arch

FILE = "data.pth"
data = torch.load(FILE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_state = data["model_state"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
tags = data['tags']
tags_unscramble = data['tags_unscramble']
patterns = data["patterns"]
responses = data["responses"]

# def printlen():
#     print(len(tags))
#     print(len(patterns))
#     print(len(responses))

# def printer(i):
#     print(tags[i])
#     print(patterns[i])
#     print(responses[i])


from transformers import pipeline, DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re

le = LabelEncoder()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
qa_model = pipeline("question-answering")


model = BERT_Arch(bert, input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


max_seq_len = 30 

def get_prediction(str):
    str = re.sub(r'[^a-zA-Z ]+', '', str)
    test_text = [str]
    model.eval()
    
    tokens_test_data = tokenizer(
    test_text,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
    )
 
    test_seq = torch.tensor(tokens_test_data['input_ids'])

    test_mask = torch.tensor(tokens_test_data['attention_mask'])
    
    preds = None
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis = 1)
    return preds[0]

def get_response(message): 
    intent = get_prediction(message)
    
    for i in range(len(patterns)): 
        # print(f"tags[{i}]: {tags[i]}")
        if tags[i] == intent:
            ans = qa_model(question = message, context = tags_unscramble[i])
            # print(f"ans: {ans}")
            score = ans['score']
            if (score > .9):
                return "Context: "+ tags_unscramble[i] + '\n\n'  + "Accuracy: " +  score + '\n' + "Response: " +  ans['answer']
            else:       
                return "Context: "+ tags_unscramble[i] + '\n\n' + "Accuracy to low: "+  score + '\n'+ "Response: "  +  ans['answer']
                
    

import gradio as gr

demo = gr.Interface(fn=get_response, inputs="text", outputs="text")

demo.launch()   