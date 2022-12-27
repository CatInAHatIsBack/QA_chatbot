import json

with open('devd.json', 'r') as f:
    myIntents = json.load(f)

data = []

for intent in myIntents['data']:
    for paragraph in intent['paragraphs']:
        context = paragraph['context'] 
        for qas in paragraph['qas']:
            question = ( qas['question'])
            # id = qas['id']
            answ = []
            if (qas['is_impossible'] == False):
                for ans in qas['answers']: 
                    answ.append(ans['text'])
                data.append([context,question,answ])
        
def getTags():
    return [array[0] for array in data]

def getPattern():
    return [array[1] for array in data]

def getResponse():
    return [array[2] for array in data]

getTags()
getPattern()
getResponse()

tag = getTags()
pattern = getTags()
response = getResponse()
def checkall():
    for i in range(1, len(tag)):
        print(f"pattern: {pattern}" )
        if tag[i] == tag[i-1]:
            print("dupe")
    
def printlen():
    print(len(tag))
    print(len(pattern))
    print(len(response))

def printer(inp):
    for i in inp:
        print(i)

printer(tag)
printlen()





