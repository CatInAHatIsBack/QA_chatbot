### install
```python
python3 -m venv env
. env/bin/activate
pip install -r ./requirements.txt
```
then run 
```python
python3 trainer.py
python3 chat.py
```

## about
project uses 2 models. 
first for getting context from your question.
second for getting answer from the context.

Traing done on the SQuAD dataset. tested with > 16k lines
data should be formatted like devd.json

## scheduler
### Without scheduler
![[Screenshot 2022-12-26 at 21.27.54.png]]![[Screenshot 2022-12-26 at 21.22.46.png]]

### with lr scheduler 
multiplicative .98 after 100
```text 
Epoch 200 / 200
train_loss: 0.003435501190578331
0.0001380878341261484
```
![[Screenshot 2022-12-27 at 13.49.45.png]]