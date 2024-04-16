import torch
from torch import nn, optim
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def train_GeoER(model, train_x, train_coord, train_n, train_y, valid_x, valid_coord, valid_n, valid_y, test_x, test_coord, test_n, test_y, device, epochs=10, batch_size=32, lr=3e-5):

  opt = optim.Adam(params=model.parameters(), lr=lr)
  criterion = nn.NLLLoss()
  
  valid_x_tensor = torch.tensor(valid_x)
  valid_coord_tensor = torch.tensor(valid_coord)
  valid_y_tensor = torch.tensor(valid_y)

  test_x_tensor = torch.tensor(test_x)
  test_coord_tensor = torch.tensor(test_coord)
  test_y_tensor = torch.tensor(test_y)

  num_steps = (len(train_x) // batch_size) * epochs
  scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)

  best_f1 = best_acc = 0.0
  best_model_state_dict = None
  for epoch in range(epochs):
    model.train()
    i = 0
    step = 1
    train_loss = 0.0

    pbar = tqdm(total=len(train_x), desc=f"Epoch {epoch+1} Train...")
    while i < len(train_x):
      
      opt.zero_grad()
      loss = torch.tensor(0.).to(device)

      if i + batch_size > len(train_x):
        y = train_y[i:]
        x = train_x[i:]
        x_coord = train_coord[i:]
        x_n = train_n[i:]
      else:
        y = train_y[i : i + batch_size]
        x = train_x[i : i + batch_size]
        x_coord = train_coord[i : i + batch_size]
        x_n = train_n[i : i + batch_size]



      y = torch.tensor(y).view(-1).to(device)
      x = torch.tensor(x)
      x_coord = torch.tensor(x_coord)
      att_mask = torch.tensor(np.where(x != 0, 1, 0))

      pred = model(x, x_coord, x_n, att_mask)

      loss = criterion(pred, y)

      loss.backward()
      opt.step()

      step += 1
      scheduler.step()
      i += batch_size
      train_loss+=loss.item()
      pbar.update(batch_size)
      pbar.set_postfix_str(f'Loss: {train_loss:.4f}')
    pbar.close()

    # print('\n*** Validation Epoch:',epoch+1,'***\n')
    this_f1,this_acc = validate_GeoER(model, valid_x_tensor, valid_coord_tensor, valid_n, valid_y_tensor, device)
    if this_f1 > best_f1:
      best_f1 = this_f1
      best_acc = this_acc
      best_model_state_dict = model.state_dict()
  if best_model_state_dict is not None:
    model.load_state_dict(best_model_state_dict)
  
  test_f1,test_acc = validate_GeoER(model, test_x_tensor, test_coord_tensor, test_n, test_y_tensor, device)
  print('\n*** Finish training ! Test F1:',test_f1 ,'on best Validation F1:',best_f1,'***\n')
  return test_f1,test_acc,best_f1,best_acc


def validate_GeoER(model, valid_x_tensor, valid_coord_tensor, valid_n, valid_y_tensor, device):

  attention_mask = np.where(valid_x_tensor != 0, 1, 0)
  attention_mask = torch.tensor(attention_mask)
  model.eval()

  acc = 0.0
  f1 = 0.0

  preds = []
  labels = []

  pbar = tqdm(total=valid_x_tensor.shape[0], desc=f"Validation...")
  for i in range(valid_x_tensor.shape[0]):
    y = valid_y_tensor[i].view(-1).to(device)
    x = valid_x_tensor[i]
    x_coord = valid_coord_tensor[i]
    x_n = valid_n[i:i+1]
    att_mask = attention_mask[i]
    pred = model(x, x_coord, x_n, att_mask, training=False)
    preds.append(torch.argmax(pred).item())
    labels.append(y.item())
    pbar.update(1)
  pbar.close()

  acc = accuracy_score(labels, preds)
  f1 = f1_score(labels, preds, pos_label=1, average='binary')

  print("Accuracy:",acc)
  print("F1-Score:",f1)
  
  return f1, acc
