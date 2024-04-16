import config
from transformers import BertModel, BertTokenizer, BertForMaskedLM
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable


class GeoUmpForPretrain(nn.Module):
  def __init__(self, device, n_emb, a_emb, dropout):
      super().__init__()
      hidden_size = config.lm_hidden
      self.language_model = BertModel.from_pretrained('bert-base-uncased')
      self.neighbert = BertModel.from_pretrained('bert-base-uncased')
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      self.ump_layer = UmpLayer(a_emb,n_emb,hidden_size,device,config.attn_type)
      self.mlm_cls = BertForMaskedLM.from_pretrained('bert-base-uncased').cls
      # self.mlm_cls = nn.Linear(hidden_size,self.tokenizer.vocab_size)
      self.linear1 = nn.Linear(hidden_size,a_emb)
      self.linear2 = nn.Linear(a_emb,2)
      self.relu = nn.ReLU()
      self.gelu = nn.GELU()
      self.tanh = nn.Tanh()
      self.leaky = nn.LeakyReLU()
      self.drop = nn.Dropout(dropout)
      self.loss_fct_mlm = nn.CrossEntropyLoss(ignore_index=-100)
      self.loss_fct_cls = nn.NLLLoss()
  def forward(self, x, x_n, att_mask,batch_size,e12pos_list,token_labels,pair_labels):
      output = self.language_model(x,att_mask)[0]
      output = self.ump_layer.inject_neighbor_sep(x_n,batch_size,output,self.neighbert,e12pos_list)
      prediction_scores = self.mlm_cls(output)
      masked_lm_loss = self.loss_fct_mlm(prediction_scores.view(-1, self.tokenizer.vocab_size), token_labels.view(-1))
      logits = F.log_softmax(self.linear2(self.drop(self.gelu(self.linear1(output[:, 0, :])))), dim=1)
      pair_cls_loss = self.loss_fct_cls(logits, pair_labels)
      return masked_lm_loss + pair_cls_loss

class UmpLayer(nn.Module):
  def __init__(self, a_emb,n_emb,hidden_size,device,attn_type):
      super().__init__()
      self.device = device
      self.attn_attr = nn.Linear(2*a_emb, 1)
      self.w_attn_attr = nn.Linear(hidden_size, a_emb)
      self.b_attn_attr = nn.Linear(1,1)
      self.query_lin = nn.Linear(hidden_size, n_emb) 
      self.key_lin = nn.Linear(hidden_size, n_emb)
      self.value_lin = nn.Linear(hidden_size, hidden_size)
      self.leaky = nn.LeakyReLU()
      self.attn_type = attn_type

  def cal_attn(self,feas,pooled_message_neighborhood,x_distances):
    x_concat = torch.cat([self.w_attn_attr(feas).view(1,-1).repeat(pooled_message_neighborhood.shape[0], 1), self.w_attn_attr(pooled_message_neighborhood)], 1)
    x_att = self.leaky(self.attn_attr(x_concat)) + self.b_attn_attr(x_distances)
    if self.attn_type=='softmax':
      x_att = F.softmax(x_att,0)
    elif self.attn_type=='sigmoid':
      x_att = x_att.sigmoid()
    elif self.attn_type=='sigmoid_relu':
      x_att = ((x_att.sigmoid() - 0.5)*2).relu()
    else:
      raise
    return x_att
        
  def inject_neighbor_sep(self,x_n,b_s,xs,neighbert,e12pos_list):
    updated_xs = []
    for b in range(b_s):
      e1_fea = xs[b,e12pos_list[b][0][0]:e12pos_list[b][0][1],:].clone()
      e2_fea = xs[b,e12pos_list[b][1][0]:e12pos_list[b][1][1],:].clone()
      e1_fea = e1_fea.unsqueeze(0) if len(e1_fea.shape)<2 else e1_fea
      e2_fea = e2_fea.unsqueeze(0) if len(e2_fea.shape)<2 else e2_fea
      e1_fea = e1_fea.unsqueeze(0) if len(e1_fea.shape)<3 else e1_fea
      e2_fea = e2_fea.unsqueeze(0) if len(e2_fea.shape)<3 else e2_fea
      x_message_neighborhood1 = []
      x_message_neighborhood2 = []
      x_distances1 = []
      x_distances2 = []
      for token_n in x_n[b]['neigh1_attr']:
        emb_n = neighbert(torch.tensor(token_n).to(self.device).unsqueeze(0))[0]
        message = F.scaled_dot_product_attention(query=self.query_lin(e1_fea), key=self.key_lin(emb_n), value=emb_n)
        x_message_neighborhood1.append(message)
      for token_n in x_n[b]['neigh2_attr']:
        emb_n = neighbert(torch.tensor(token_n).to(self.device).unsqueeze(0))[0]
        message = F.scaled_dot_product_attention(query=self.query_lin(e2_fea), key=self.key_lin(emb_n), value=emb_n)
        x_message_neighborhood2.append(message)

      if not len(x_n[b]['dist1']):
        x_distances1.append(1000)
      else:
        x_distances1 =  x_n[b]['dist1']
      if not len(x_n[b]['dist2']):
        x_distances2.append(1000)
      else:
        x_distances2 =  x_n[b]['dist2']
      x_distances1 = torch.tensor(x_distances1, dtype=torch.float).view(-1, 1).to(self.device)
      x_distances2 = torch.tensor(x_distances2, dtype=torch.float).view(-1, 1).to(self.device)
      if len(x_message_neighborhood1):
        pooled_message_neighborhood1 = torch.cat([e.mean(1) for e in x_message_neighborhood1],dim=0).to(self.device)
      else:
        pooled_message_neighborhood1 = torch.zeros((1,768)).to(self.device)
        x_message_neighborhood1.append(torch.ones((1,1,768)).to(self.device))
      x_att1 = self.cal_attn(e1_fea.mean(dim=1),pooled_message_neighborhood1,x_distances1)

      if len(x_message_neighborhood2):
        pooled_message_neighborhood2 = torch.cat([e.mean(1) for e in x_message_neighborhood2],dim=0).to(self.device)
      else:
        pooled_message_neighborhood2 = torch.zeros((1,768)).to(self.device)
      x_att2 = self.cal_attn(e2_fea.mean(dim=1),pooled_message_neighborhood2,x_distances2)
      x = xs[b].unsqueeze(0).clone()
      for score,message in zip(x_att1,x_message_neighborhood1):
        x[:,e12pos_list[b][0][0]:e12pos_list[b][0][1],:]+=score*message
      for score,message in zip(x_att2,x_message_neighborhood2):
        x[:,e12pos_list[b][1][0]:e12pos_list[b][1][1],:]+=score*message
      updated_x = F.scaled_dot_product_attention(query=self.query_lin(x), key=self.key_lin(x), value=self.value_lin(x))
      updated_xs.append(updated_x.squeeze())
    updated_xs = torch.stack(updated_xs)
    return updated_xs
  
  def inject_neighbor(self,x_n,b_s,xs,neighbert):
    raise
    pass
    # pooled_xs = []
    # for b in range(b_s):
    #   x_message_neighborhood = []
    #   for token_n in x_n[b]['neigh1_attr']:
    #     emb_n = neighbert(torch.tensor(token_n).to(self.device).unsqueeze(0))[0]
    #     message = F.scaled_dot_product_attention(query=self.query_lin(xs[b].unsqueeze(0)), key=self.key_lin(emb_n), value=emb_n)
    #     x_message_neighborhood.append(message)
    #   for token_n in x_n[b]['neigh2_attr']:
    #     emb_n = neighbert(torch.tensor(token_n).to(self.device).unsqueeze(0))[0]
    #     message = F.scaled_dot_product_attention(query=self.query_lin(xs[b].unsqueeze(0)), key=self.key_lin(emb_n), value=emb_n)
    #     x_message_neighborhood.append(message)
    #   x_distances =  x_n[b]['dist1']+ x_n[b]['dist2']
    #   if not len(x_distances):
    #     x_distances.append(1000)
    #   x_distances = torch.tensor(x_distances, dtype=torch.float).view(-1, 1).to(self.device)

    #   if len(x_message_neighborhood):
    #     x_message_neighborhood = torch.cat(x_message_neighborhood,dim=0).to(self.device)
    #   else:
    #     x_message_neighborhood = torch.zeros((1,1,768)).to(self.device)

    #   pooled_message_neighborhood = x_message_neighborhood.mean(dim=1)
    #   x_concat = torch.cat([self.w_attn_attr(xs[b].mean(dim=0)).view(1,-1).repeat(pooled_message_neighborhood.shape[0], 1), self.w_attn_attr(pooled_message_neighborhood)], 1)
    #   x_att = self.leaky(self.attn_attr(x_concat)) + self.b_attn_attr(x_distances)
      
    #   if self.attn_type=='softmax':
    #     x_att = F.softmax(x_att,0)
    #   elif self.attn_type=='sigmoid':
    #     x_att = x_att.sigmoid()
    #   elif self.attn_type=='sigmoid_relu':
    #     x_att = ((x_att.sigmoid() - 0.5)*2).relu()
    #   else:
    #     raise
    #   x=(xs[b] + torch.sum(x_att.unsqueeze(-1)*x_message_neighborhood,0)).unsqueeze(0)
    #   pooled_x = F.scaled_dot_product_attention(query=self.query_lin(x), key=self.key_lin(x), value=self.value_lin(x))[:, 0, :]
    #   pooled_xs.append(pooled_x.squeeze())
    # pooled_xs = torch.stack(pooled_xs)
    # return pooled_xs
  
class GeoUmpER(nn.Module):
  def __init__(self, device, c_emb, n_emb, a_emb, dropout, finetuning=True):
      super().__init__()

      hidden_size = config.lm_hidden

      self.language_model = BertModel.from_pretrained('bert-base-uncased')
      self.neighbert = BertModel.from_pretrained('bert-base-uncased')
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

      self.device = device
      self.finetuning = finetuning

      self.drop = nn.Dropout(dropout)
      self.attn = nn.Linear(hidden_size, 1)
      self.linear1 = nn.Linear(hidden_size + 2*c_emb + n_emb, (hidden_size + 2*c_emb + n_emb)//2)
      self.linear2 = nn.Linear((hidden_size + 2*c_emb + n_emb)//2, 2)

      self.neigh_linear = nn.Linear(2*a_emb, n_emb)
      self.coord_linear = nn.Linear(1, 2*c_emb)

      self.attn = nn.Linear(2*a_emb, 1)
      self.w_attn = nn.Linear(hidden_size, a_emb)
      self.b_attn = nn.Linear(1,1)

      if config.use_ump:
        self.ump_layer = UmpLayer(a_emb,n_emb,hidden_size,device,config.attn_type)

      self.relu = nn.ReLU()
      self.gelu = nn.GELU()
      self.tanh = nn.Tanh()
      self.leaky = nn.LeakyReLU()

  def encode_neighbor(self,x_n,b_s):
    x_neighbors = []
    for b in range(b_s):
      x_neighborhood1 = []
      x_neighborhood2 = []
      x_distances1 = []
      x_distances2 = []
      with torch.no_grad():
        x_node1 = self.neighbert(torch.tensor(x_n[b]['name1']).to(self.device).unsqueeze(0))[0]
        x_node2 = self.neighbert(torch.tensor(x_n[b]['name2']).to(self.device).unsqueeze(0))[0]
      
        for x_n1 in x_n[b]['neigh1']:
          emb_n1 = self.neighbert(torch.tensor(x_n1).to(self.device).unsqueeze(0))[0]
          x_neighborhood1.append(torch.mean(emb_n1,dim=1).squeeze())
        for x_n2 in x_n[b]['neigh2']:
          emb_n2 = self.neighbert(torch.tensor(x_n2).to(self.device).unsqueeze(0))[0]
          x_neighborhood2.append(torch.mean(emb_n2,dim=1).squeeze())

        if not len(x_n[b]['dist1']):
          x_distances1.append(1000)
        else:
          x_distances1 = x_n[b]['dist1']
        if not len(x_n[b]['dist2']):
          x_distances2.append(1000)
        else:
          x_distances2 = x_n[b]['dist2']
        if not len(x_neighborhood1):
          x_neighborhood1.append(torch.zeros(768))
        if not len(x_neighborhood2):
          x_neighborhood2.append(torch.zeros(768))

        x_neighborhood1 = torch.stack(x_neighborhood1).to(self.device)
        x_neighborhood2 = torch.stack(x_neighborhood2).to(self.device)
        x_distances1 = torch.tensor(x_distances1, dtype=torch.float).view(-1, 1).to(self.device)
        x_distances2 = torch.tensor(x_distances2, dtype=torch.float).view(-1, 1).to(self.device)

        avg_x_node1 = torch.mean(x_node1,dim=1)
        avg_x_node2 = torch.mean(x_node2,dim=1)


      x_concat1 = torch.cat([self.w_attn(avg_x_node1).view(1,-1).repeat(x_neighborhood1.shape[0], 1), self.w_attn(x_neighborhood1)], 1)
      x_concat2 = torch.cat([self.w_attn(avg_x_node2).view(1,-1).repeat(x_neighborhood2.shape[0], 1), self.w_attn(x_neighborhood2)], 1)
      x_att1 = F.softmax(self.leaky(self.attn(x_concat1)) + self.b_attn(x_distances1),0)
      x_att2 = F.softmax(self.leaky(self.attn(x_concat2)) + self.b_attn(x_distances2),0)
      x_context1 = torch.sum(self.w_attn(x_neighborhood1)*x_att1,0)
      x_context2 = torch.sum(self.w_attn(x_neighborhood2)*x_att2,0)
      x_sim1 = x_context1*self.w_attn(avg_x_node1).squeeze()
      x_sim2 = x_context2*self.w_attn(avg_x_node2).squeeze()

      x_neighbors.append(self.relu(torch.cat([x_sim1, x_sim2])))

    x_neighbors = torch.stack(x_neighbors)
    x_neighbors = self.neigh_linear(x_neighbors)
    return x_neighbors

  def forward(self, x, x_coord, x_n, att_mask, training=True):
    x = x.to(self.device)
    att_mask = att_mask.to(self.device)
    x_coord = x_coord.to(self.device)
    self.neighbert.eval()

    if len(x.shape) < 2:
      x = x.unsqueeze(0)

    if len(att_mask.shape) < 2:
      att_mask = att_mask.unsqueeze(0)

    while len(x_coord.shape) < 2:
      x_coord = x_coord.unsqueeze(0)

    b_s = x.shape[0]
    if training and self.finetuning:
      self.language_model.train()
      self.train()
      output = self.language_model(x, attention_mask=att_mask)[0]

    else:
      self.language_model.eval()
      with torch.no_grad():
        output = self.language_model(x, attention_mask=att_mask)[0]

    if config.use_neighbor:
      x_neighbors = self.encode_neighbor(x_n,b_s)
    else:
      x_neighbors = torch.zeros(b_s,config.n_em).to(self.device)

    if config.use_ump:
      if config.inject_sep:
        e12pos_list = []
        for subx in x:
            all_sep = torch.where(subx==102)[0].tolist()
            if len(all_sep)==2:
              sep1,sep2 = all_sep
            else:
              sep2 = all_sep[-1]
              for ind in all_sep[:-1]:
                if subx[ind+1].item()==8902:
                  sep1 = ind
                  break
            e12pos_list.append(
              ([1,sep1-1],[sep1+1,sep2-1])
            )
        pooled_output = self.ump_layer.inject_neighbor_sep(x_n,b_s,output,self.neighbert,e12pos_list)[:, 0, :]
      else:
        pooled_output = self.ump_layer.inject_neighbor(x_n,b_s,output,self.neighbert)
    else:
      pooled_output = output[:, 0, :] # take only 0 (the position of the [CLS])
  
    x_coord = x_coord.transpose(0,1)
    x_coord = self.coord_linear(x_coord)

    output = torch.cat([pooled_output, x_coord, x_neighbors], 1)

    output = self.linear2(self.drop(self.gelu(self.linear1(output))))
    
    return F.log_softmax(output, dim=1)
