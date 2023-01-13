import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import io
import os
import numpy as np
import pandas as pd
import PIL.Image
import pytorch_lightning as pl
import torch
from IPython.display import Image, display
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from transformers import (AdamW, BertModel, BertTokenizerFast,
                          VisualBertForQuestionAnswering, ViTModel,
                          get_linear_schedule_with_warmup)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os
import warnings

import pytorch_lightning as pl
import torch
from mmt_retrieval import MultimodalTransformer
from mmt_retrieval.model.models import (M3P, OSCAR, UNITER, ClassificationHead,
                                        Pooling)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
pretrained_model = M3P("XXX/M3P/",max_seq_length=128)
class_head = ClassificationHead(num_labels=2)
model =MultimodalTransformer(modules=[pretrained_model, class_head])
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss

data_dir="XXX/M3P/Meme_my_data_roman_explain_primary_emotion.csv"
data=pd.read_csv(data_dir)
len(data)

import glob
import os

import tqdm

outliers = []
data=pd.read_csv(data_dir)
for names in list(data['Name']):
  if not os.path.exists('XXX/code/My_annotation_guideline/roman_meme/my_roman_actual/'+names):
    outliers.append(names)
images = list(map(lambda x: x.split('.')[0], list(pd.read_csv(data_dir)['Name'])))
sent = list(pd.read_csv(data_dir)['Text'])
caption = list(pd.read_csv(data_dir)['Explainantion'])
label = list(map(lambda x: int(x), list(pd.read_csv(data_dir)['Level1'])))
inten = list(map(lambda x: int(x), list(pd.read_csv(data_dir)['Level2'])))
folder = 'XXX/M3P/my_roman_meme_image_feature/'
for filename in glob.iglob(os.path.join(folder, '*.npy')):
    os.rename(filename, filename[:-4] + '.npz')
for i in images:
  act = np.load(os.path.join(folder,i+'.npz'),allow_pickle=True)
  res = np.load(os.path.join(folder,i+'_info.npz'),allow_pickle=True).item()
  
  data = {"features": act, "boxes": res["bbox"], "num_boxes": res["num_boxes"],
                              "img_h": res["image_height"], "img_w": res["image_width"]}
  np.savez_compressed(os.path.join('XXX/M3P/features_img',i+'.npz'), x=act, bbox=res['bbox'], num_bbox = res['num_boxes'], image_h = res['image_height'], image_w = res['image_width'])
model.image_dict.load_file_names('XXX/M3P/features_img')
{key.split(".")[0]: os.path.join('XXX/M3P/', key) for key in os.listdir('XXX/M3P/features_img')}
def chunks(l, n):
    n_items = len(l)
    if n_items % n:
        n_pads = n - (n_items % n)
    else:
        n_pads = 0
    l = l + ['<PAD>' for _ in range(n_pads)] 
    for i in range(0, len(l), n):
        yield l[i:i + n]

def process(idx_val,arr):
  if idx_val=='0':
    arr.append(0)
  else:
    arr.append(1)
import pandas as pd

outliers
import torch
import torch.nn as nn
import torch.nn.functional as F

class MFB(nn.Module):
    def __init__(self,img_feat_size, ques_feat_size, is_first, MFB_K, MFB_O, DROPOUT_R):
        super(MFB, self).__init__()
        
        self.MFB_K = MFB_K
        self.MFB_O = MFB_O
        self.DROPOUT_R = DROPOUT_R
        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)
        
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride = MFB_K)
    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                
        ques_feat = self.proj_q(ques_feat)              
        
        exp_out = img_feat * ques_feat             
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     
        z = self.pool(exp_out) * self.MFB_K         
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         
        z = z.view(batch_size, -1, self.MFB_O)      
        return z
import os
import warnings

import pytorch_lightning as pl
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

dataset_path="XXX/M3P/Meme_my_data_roman_explain_primary_emotion.csv"
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)

def get_data_code_mixed(dataset_path):
  name, text_features,image_features,l,a,v, fe, Neg, neg,ir, ra, disg, ner, sh, disa, en, su, sa, jo, pr, sar, hum, inten= [],[],[],[],[], \
  [],[],[],[],[],[],[],[],[],[],[],[],[], [],[],[],[],[]
  k = pd.read_csv(dataset_path)
  text = list(k['Text'])
  dataframe = k.apply(LabelEncoder().fit_transform)
  images = list(map(lambda x: x.split('.')[0], list(k['Name'])))
  images = list(map(lambda x: x.split('.')[0], list(k['Name'])))
  k['Text'] = k['Text'].str.split().apply(lambda x: list(chunks(x, 128)))
  k = k.explode('Text').reset_index(drop=True)
  k['Text'] = k['Text'].apply(' '.join)
  sent = list(k['Text'])
  print(len(sent))
  name = list(k['Name'])
  k['Explainantion'] = k['Explainantion'].str.split().apply(lambda x: list(chunks(x, 128)))
  k = k.explode('Explainantion').reset_index(drop=True)
  k['Explainantion'] = k['Explainantion'].apply(' '.join)
  cap = list(k['Explainantion'])
  label = list(map(lambda x: int(x), list(k['Level1'])))
  inten = list(map(lambda x: int(x), list(k['Level2'])))
  valence = list(k['Valence_2_3_add'])
  valence = list(map(lambda x: x - 1 , valence))
  arousal = list(k['Arousal'])
  arousal = list(map(lambda x: x - 1 , arousal))
  Fear = list(k['Fear'])   
  Rage	= list(k['Anger'])   
  Surprise	= list(k['Surprise']) 
  Sadness	= list(k['Sadness'])
  Joy	= list(k['Joy'])  
  
  Sarcasm = list(k['Sarcasm_0_1_add']) 
  Humor = list(k['Humor'])  
  sentence_embeddings = model.encode(sentences=sent,output_value ='token_embeddings', convert_to_numpy=True)
  caption_embeddings = model.encode(sentences=cap,output_value ='token_embeddings', convert_to_numpy=True)
  model.image_dict.load_file_names('XXX/M3P/features_img')
  image_embeddings = model.encode(images=images,output_value='token_embeddings', convert_to_numpy=True)
  return sentence_embeddings,image_embeddings,caption_embeddings,label,arousal,valence, Fear, Rage, Surprise,	Sadness, Joy, Sarcasm, Humor, \
               inten,name
class code_mixed(Dataset):
  def __init__(self,dataset_path):
    
    self.t_f,self.i_f,self.cap,self.label,self.a,self.v,self.fe, self.ra, self.su, self.sa, self.jo, \
    self.sar, self.hum, self.inten, self.name= get_data_code_mixed(dataset_path)
    self.t_f = np.asarray(self.t_f)
    self.i_f = np.asarray(self.i_f)
    self.cap = np.asarray(self.cap)
    
  def __len__(self):
    return len(list(self.label))
  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    T = self.t_f[idx]
    I = self.i_f[idx]
    C = self.cap[idx]
    label = self.label[idx]
    v = self.v[idx]
    a = self.a[idx]
    fe = self.fe[idx]
    ra = self.ra[idx]
    su = self.su[idx] 
    sa = self.sa[idx]
    jo = self.jo[idx]
    
    sar = self.sar[idx]
    hum = self.hum[idx]
    inten = self.inten[idx]
    name = self.name[idx]
    sample = {'processed_txt':T,'processed_img':I,'processed_cap':C,'label':label,'arousal':a ,'valence':v, 'fear': fe,\
       'rage':ra, 'surprise':su, 'sadness':sa, 'joy':jo,'sarcasm':sar, 'humor': hum, \
              'inten': inten, 'name':name}
    return sample

a=code_mixed("XXX/M3P/Meme_my_data_roman_explain_primary_emotion.csv")

len(a)
torch.manual_seed(432)
t_p,te_p = torch.utils.data.random_split(a,[6000,967])
t_p,v_p = torch.utils.data.random_split(t_p,[5500,500])

pred_e = 0
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

class Classifier(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.MFB = MFB(768,768,True,256,64,0.1)
    self.x_flatten=torch.nn.Linear(99840,768)
    self.y_flatten=torch.nn.Linear(39168,768)
    self.flatten=torch.nn.Flatten()
    self.loss_fn_emotion=torch.nn.KLDivLoss(reduction='batchmean',log_target=True)
    self.encode_text = torch.nn.Linear(1280,64)
    self.fin = torch.nn.Linear(64,2)
    self.flatten=torch.nn.Flatten()
    self.fin_inten = torch.nn.Linear(64,3)
    self.mask= torch.tensor([0, 1]).cuda()
  def forward(self, x,y,off_label):
      
      x_,y_ = x,y
      x = x.float()
      y = y.float()
      x=self.flatten(x)
      y=self.flatten(y)
      x=self.x_flatten(x)
      y=self.y_flatten(y)
      z_ = self.MFB(torch.unsqueeze(y,axis=1),torch.unsqueeze(x,axis=1))
      z = z_
      c = self.fin(torch.squeeze(z,dim=1))
      for i in range(len(off_label)):
        if (off_label[i]==1):
          c_inten = self.mask[1]*(self.fin_inten(torch.squeeze(z,dim=1)))
        else: c_inten = self.mask[0]*(self.fin_inten(torch.squeeze(z,dim=1)))
      print("This time c_inten",c_inten)
      c = torch.log_softmax(c, dim=1)
      c_inten = torch.log_softmax(c_inten, dim=1)
      return z,c, c_inten
  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)
  
  def training_step(self, train_batch, batch_idx):
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity,name= train_batch
      
      lab = train_batch[lab]
      txt = train_batch[txt]
      img = train_batch[img]
      intensity = train_batch[intensity]
      z,logit_offen,inten = self.forward(txt,img,lab) 
            
      inten_levels = levels_from_labelbatch(
            intensity, num_classes=3).type_as(inten)
      loss1 = self.cross_entropy_loss(logit_offen, lab)
      tmp = np.argmax(logit_offen.detach().cpu().numpy(),axis=-1)
      for i in range(len(tmp)):
        if (tmp[i]==1):
          loss_int = self.cross_entropy_loss(inten, intensity)
          loss=loss1+loss_int
        else: loss=loss1
      self.log('train_loss', loss)
      return loss
  def validation_step(self, val_batch, batch_idx):
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity,name= val_batch
      lab = val_batch[lab]
      txt = val_batch[txt]
      img = val_batch[img]
      intensity = val_batch[intensity]
      _,logits,inten = self.forward(txt,img,lab)   
      
      inten_levels = levels_from_labelbatch(
            intensity, num_classes=3).type_as(inten)
      
      
      
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      
      
      loss1 = self.cross_entropy_loss(logits, lab)
      for i in range(len(tmp)):
        if (tmp[i]==1):
          
          loss_int = self.cross_entropy_loss(inten, intensity)
          loss=loss1+loss_int
        else: loss=loss1
      
      lab = lab.detach().cpu().numpy()
      self.log('val_acc', f1_score(lab,tmp,average='macro'))
      
      self.log('val_loss', loss)
      tqdm_dict = {'val_acc': accuracy_score(lab,tmp)}
      
      return {
                'progress_bar': tqdm_dict,  
      
      'val_acc intensity':f1_score(intensity.detach().cpu().numpy(),np.argmax(inten.detach().cpu().numpy(),axis=-1),average='macro')
      }
      
  def validation_epoch_end(self, validation_step_outputs):
    outs = []
    outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14,outs16,outs17 = \
    [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    outs15 = []
    outs18 = []
    for out in validation_step_outputs:
      outs.append(out['progress_bar']['val_acc'])
      outs14.append(out['val_acc intensity'])
    self.log('val_acc_all_offn', sum(outs)/len(outs))
    self.log('val_acc_all inten', sum(outs14)/len(outs14))
    print(f'***offensive f1 at epoch end {sum(outs)/len(outs)}****')
    print(f'***val acc inten at epoch end {sum(outs14)/len(outs14)}****')
  def test_step(self, batch, batch_idx):
      
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity, name=batch
      lab = batch[lab]
      txt = batch[txt]
      img = batch[img]
      name = batch[name]
      intensity = batch[intensity]
      _,logits,inten = self.forward(txt,img,lab)
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      tmp_int = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss1 = self.cross_entropy_loss(logits, lab)
      inten_levels = levels_from_labelbatch(
            intensity, num_classes=3).type_as(inten)
      for i in range(len(tmp)):
        if (tmp[i]==1):
          
          loss_int = self.cross_entropy_loss(inten, intensity)
          loss=loss1+loss_int
        else: loss=loss1
      lab = lab.detach().cpu().numpy()
      
      f1_intensity = f1_score(intensity.detach().cpu().numpy(),np.argmax(inten.detach().cpu().numpy(),axis=-1),average='macro')
      
      self.log('test_acc', accuracy_score(lab,tmp))
      np.save('actual_label_offensive_before_explain.npy',lab)
      np.save('predicted_label_offensive_before_explain.npy',tmp)
      
      np.save('name_before_explain.npy',name)
      
      np.save('predicted_label_intensity_before_explain.npy',tmp_int)
      print(f'confusion matrix intensity {confusion_matrix(intensity.detach().cpu().numpy(),np.argmax(inten.detach().cpu().numpy(),axis=-1))}')
      print(f'confusion matrix offensive {confusion_matrix(lab,tmp)}')
      best_threshold = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
      return {'test_loss': loss,
              
              
               'test_off f1_score':f1_score(lab,tmp,average='macro'),
              'test_off acc_sco': accuracy_score(lab,tmp),
              'test_acc intensity': accuracy_score(intensity.detach().cpu().numpy(),np.argmax(inten.detach().cpu().numpy(),axis=-1)),
              'test_f1 inten': f1_score(intensity.detach().cpu().numpy(),np.argmax(inten.detach().cpu().numpy(),axis=-1),average='macro')
              }
  def test_epoch_end(self, outputs):
        
        outs = []
        outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14 = \
        [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        outs15 = []
        outs16 = []
        outs17 = []
        outs18 = []
        for out in outputs:
          outs.append(out['test_off f1_score'])
          outs14.append(out['test_off acc_sco'])
          outs16.append(out['test_acc intensity'])
          outs17.append(out['test_f1 inten'])
        self.log('test_off f1_score', sum(outs)/len(outs))
        self.log('test_off acc_sco', sum(outs14)/len(outs14))
        self.log('test_acc intensity', sum(outs16)/len(outs16))
        self.log('test_f1 inten', sum(outs17)/len(outs17)) 
        
  def configure_optimizers(self):
    
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
    return optimizer

class HmDataModule(pl.LightningDataModule):
  def setup(self, stage):
    
      
    
    self.hm_train = t_p
    self.hm_val = v_p
    self.hm_test = te_p
    
  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=64, drop_last=True)
  def val_dataloader(self):
    return DataLoader(self.hm_val, batch_size=64, drop_last=True)
  def test_dataloader(self):
    return DataLoader(self.hm_test, batch_size=128)

data_module = HmDataModule()

checkpoint_callback = ModelCheckpoint(
    
     monitor='val_acc_all inten',  
     dirpath='ckpts_baseline_coral/',
     filename='masking_model{epoch:02d}-val_acc{val_acc_all inten:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="max",
 )
all_callbacks = []
all_callbacks.append(checkpoint_callback)
from pytorch_lightning import seed_everything

seed_everything(seed=1234, workers=True)
hm_model = Classifier()
gpus = 1 if torch.cuda.is_available() else 0
trainer = pl.Trainer(gpus=gpus,max_epochs=60,callbacks=all_callbacks)
trainer.fit(hm_model, data_module)
Classifier_baseline = Classifier.load_from_checkpoint('XXX//masking_model18-val_acc0.41.ckpt')
Classifier_baseline.to(device)
test_dataloader = DataLoader(dataset=te_p, batch_size=1478)
ckpt_path = 'XXX//masking_model18-val_acc0.41.ckpt' 
trainer.test(dataloaders=test_dataloader,ckpt_path=ckpt_path)
pred_e = 0
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
class Classifier_hate_explain(pl.LightningModule):
  def __init__(self):
    super().__init__()
    
    
    self.MFB = MFB(768,768,True,256,64,0.1)
    self.loss_fn_emotion=torch.nn.KLDivLoss(reduction='batchmean',log_target=True)
    
    self.encode_text = torch.nn.Linear(1280,64)
    self.x_flatten=torch.nn.Linear(99840,768)
    self.y_flatten=torch.nn.Linear(39168,768)
    self.flatten=torch.nn.Flatten()
    self.fin_v = CoralLayer(size_in=64, num_classes=4)
    self.fin_sarcasm = torch.nn.Linear(64,3)
    self.fin_e1 = torch.nn.Linear(64,2)
    self.fin_e2 = torch.nn.Linear(64,2)
    self.fin_e3 = torch.nn.Linear(64,2)
    self.fin_e4 = torch.nn.Linear(64,2)
    self.fin_e5 = torch.nn.Linear(64,2)
    
  def forward(self, x,y):
      
      x_,y_ = x,y
      x = x.float()
      y = y.float()
      
      x=self.flatten(x)
      y=self.flatten(y)
      x=self.x_flatten(x)
      y=self.y_flatten(y)
      z_ = self.MFB(torch.unsqueeze(y,axis=1),torch.unsqueeze(x,axis=1))
      z = z_
      c_v = self.fin_v(torch.squeeze(z,dim=1))
      
      c_e1 = self.fin_e1(torch.squeeze(z,dim=1))
      c_e2 = self.fin_e2(torch.squeeze(z,dim=1))
      c_e3 = self.fin_e3(torch.squeeze(z,dim=1))
      c_e4 = self.fin_e4(torch.squeeze(z,dim=1))
      c_e5 = self.fin_e5(torch.squeeze(z,dim=1))
      c_sarcasm = self.fin_sarcasm(torch.squeeze(z,dim=1))
      c_e1 = torch.log_softmax(c_e1, dim=1)
      c_sarcasm = torch.log_softmax(c_sarcasm, dim=1)
      c_e2 = torch.log_softmax(c_e2, dim=1)
      c_e3 = torch.log_softmax(c_e3, dim=1)
      c_e4 = torch.log_softmax(c_e4, dim=1)
      c_e5 = torch.log_softmax(c_e5, dim=1)
      return z,c_v,c_e1,c_e2,c_e3,c_e4,c_e5, c_sarcasm
  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)
  def training_step(self, train_batch, batch_idx):
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity, name= train_batch
      lab = train_batch[lab]
      txt = train_batch[txt]
      img = train_batch[img]
      val = train_batch[val]
      e1 = train_batch[e1]
      e2 = train_batch[e2]
      e3 = train_batch[e3]
      e4 = train_batch[e4]
      e5 = train_batch[e5]
      sarcasm = train_batch[sarcasm]
      z,logit_val,a,b,c,d,e,logit_sarcasm= self.forward(txt,img) 
      
      val_level = levels_from_labelbatch(
            val, num_classes=4).type_as(logit_val)
      loss3  = coral_loss(logit_val, val_level)
      loss4 = self.cross_entropy_loss(a, e1)
      loss5 = self.cross_entropy_loss(b, e2)
      loss6 = self.cross_entropy_loss(c, e3)
      loss7 = self.cross_entropy_loss(d, e4)
      loss8 = self.cross_entropy_loss(e, e5)
      loss_sarcasm = self.cross_entropy_loss(logit_sarcasm, sarcasm)
      loss=loss3+loss5+loss7+loss6+loss4+loss8+loss_sarcasm
      self.log('train_loss', loss)
      
      return loss
  def validation_step(self, val_batch, batch_idx):
      
      
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity, name= val_batch
      
      
      txt = val_batch[txt]
      img = val_batch[img]
      val = val_batch[val]
      
      e1 = val_batch[e1]
      e2 = val_batch[e2]
      e3 = val_batch[e3]
      e4 = val_batch[e4]
      e5 = val_batch[e5]
      sarcasm = val_batch[sarcasm]
      z,logit_val,a,b,c,d,e,logit_sarcasm = self.forward(txt,img)
      val_level = levels_from_labelbatch(val, num_classes=4).type_as(logit_val)
      tmp=proba_to_label(torch.sigmoid(logit_val)).detach().cpu().numpy()
      loss  = coral_loss(logit_val, val_level)
      val = val.detach().cpu().numpy()
      self.log('val_acc', f1_score(val,tmp,average='macro'))
      self.log('val_loss', loss)
      tqdm_dict = {'val_acc': accuracy_score(val,tmp)}
      return {
                'progress_bar': tqdm_dict,
              'val_acc e1': accuracy_score(e1.detach().cpu().numpy(),np.argmax(a.detach().cpu().numpy(),axis=-1)),
      'val_acc e2': accuracy_score(e2.detach().cpu().numpy(),np.argmax(b.detach().cpu().numpy(),axis=-1)),
      'val_acc e3': accuracy_score(e3.detach().cpu().numpy(),np.argmax(c.detach().cpu().numpy(),axis=-1)),
      'val_acc e4': accuracy_score(e4.detach().cpu().numpy(),np.argmax(d.detach().cpu().numpy(),axis=-1)),
      'val_acc e5': accuracy_score(e5.detach().cpu().numpy(),np.argmax(e.detach().cpu().numpy(),axis=-1)),
      'val_acc valence': f1_score(val,tmp,average='macro'),
       'val_acc sarcasm': accuracy_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1)),
       'f1 sarcasm': f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro')
      }
      
  def validation_epoch_end(self, validation_step_outputs):
    outs = []
    outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14,outs16,outs17 = \
    [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    outs15 = []
    outs18 = []
    for out in validation_step_outputs:
      outs.append(out['progress_bar']['val_acc'])
      outs1.append(out['val_acc e1'])
      outs2.append(out['val_acc e2'])
      outs3.append(out['val_acc e3'])
      outs4.append(out['val_acc e4'])
      outs5.append(out['val_acc e5'])
      outs14.append(out['val_acc valence'])
      outs17.append(out['val_acc sarcasm'])
      outs18.append(out['f1 sarcasm'])
    self.log('val_acc_all_offn', sum(outs)/len(outs))
    self.log('val_acc_all e1', sum(outs1)/len(outs1))
    self.log('val_acc_all e2', sum(outs2)/len(outs2))
    self.log('val_acc_all e3', sum(outs3)/len(outs3))
    self.log('val_acc_all e4', sum(outs4)/len(outs4))
    self.log('val_acc_all e5', sum(outs5)/len(outs5))
    self.log('val_acc_all valence', sum(outs14)/len(outs14))
    self.log('val_acc_all sarcasm', sum(outs17)/len(outs17))
    self.log('val_f1_all sarcasm', sum(outs18)/len(outs18))
    print(f'***valence f1 at epoch end {sum(outs)/len(outs)}****')
    print(f'***val acc valence at epoch end {sum(outs14)/len(outs14)}****')
    print(f'***val_acc_all e1 at epoch end {sum(outs1)/len(outs1)}****')
    print(f'***val_acc_all e2 at epoch end {sum(outs2)/len(outs2)}****')
    print(f'***val_acc_all e3 at epoch end {sum(outs3)/len(outs3)}****')
    print(f'***val_acc_all e4 at epoch end {sum(outs4)/len(outs4)}****')
    print(f'***val_acc_all e5 at epoch end {sum(outs5)/len(outs5)}****')
    print(f'***val acc sarcasm at epoch end {sum(outs17)/len(outs17)}****')
    print(f'***val f1 sarcasm at epoch end {sum(outs18)/len(outs18)}****')
  def test_step(self, batch, batch_idx):
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity, name= batch
      val = batch[val]
      txt = batch[txt]
      img = batch[img]
      e1 = batch[e1]
      e2 = batch[e2]
      e3 = batch[e3]
      e4 = batch[e4]
      e5 = batch[e5]
      sarcasm = batch[sarcasm]
      z,logit_val,a,b,c,d,e,logit_sarcasm = self.forward(txt,img)
      tmp=proba_to_label(torch.sigmoid(logit_val)).detach().cpu().numpy()
      val_level = levels_from_labelbatch(val, num_classes=4).type_as(logit_val)
      loss  = coral_loss(logit_val, val_level)
      val = val.detach().cpu().numpy()
      self.log('test_acc', accuracy_score(val,tmp))
      self.log('test f1',f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro'))
      print(f'confusion matrix intensity {confusion_matrix(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1))}')
      print(f'confusion matrix offensive {confusion_matrix(val,tmp)}')
      self.log('test_loss', loss)
      return {'test_loss': loss,
              'test_acc': accuracy_score(val,tmp),
              'test_acc e1': accuracy_score(e1.detach().cpu().numpy(),np.argmax(a.detach().cpu().numpy(),axis=-1)),
              'test_acc e2': accuracy_score(e2.detach().cpu().numpy(),np.argmax(b.detach().cpu().numpy(),axis=-1)),
              'test_acc e3': accuracy_score(e3.detach().cpu().numpy(),np.argmax(c.detach().cpu().numpy(),axis=-1)),
              'test_acc e4': accuracy_score(e4.detach().cpu().numpy(),np.argmax(d.detach().cpu().numpy(),axis=-1)),
              'test_acc e5': accuracy_score(e5.detach().cpu().numpy(),np.argmax(e.detach().cpu().numpy(),axis=-1)),
              'test_acc sarcasm': accuracy_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1)),
              'f1 sarcasm': f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro')}
  def test_epoch_end(self, outputs):
        
        outs = []
        outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14 = \
        [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        outs15 = []
        outs16 = []
        outs17 = []
        outs18 = []
        for out in outputs:
          outs.append(out['test_acc'])
          outs1.append(out['test_acc e1'])
          outs2.append(out['test_acc e2'])
          outs3.append(out['test_acc e3'])
          outs4.append(out['test_acc e4'])
          outs5.append(out['test_acc e5'])
          outs16.append(out['test_acc sarcasm'])
          
          outs18.append(out['f1 sarcasm'])
        
        self.log('final test f1', sum(outs)/len(outs))
        self.log('test_acc_all e1', sum(outs1)/len(outs1))
        self.log('test_acc_all e2', sum(outs2)/len(outs2))
        self.log('test_acc_all e3', sum(outs3)/len(outs3))
        self.log('test_acc_all e4', sum(outs4)/len(outs4))
        self.log('test_acc_all e5', sum(outs5)/len(outs5))
        self.log('test_acc_all sarcasm', sum(outs16)/len(outs16))
        self.log('test_f1_all sarcasm', sum(outs18)/len(outs18))
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
    return optimizer

class HmDataModule(pl.LightningDataModule):
  def setup(self, stage):
    self.hm_train = t_p
    self.hm_val = v_p
    self.hm_test = te_p
  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=64)
  def val_dataloader(self):
    return DataLoader(self.hm_val, batch_size=64)
  def test_dataloader(self):
    return DataLoader(self.hm_test, batch_size=128)

data_module = HmDataModule()

checkpoint_callback = ModelCheckpoint(
     monitor='val_acc_all_offn',
     dirpath='ckpts_roman_meme_text_explain_data/',
     filename='meme_text-explain{epoch:02d}-val_f1_all_offn{val_acc_all_offn:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="max",
 )
all_callbacks = []
all_callbacks.append(checkpoint_callback)
from pytorch_lightning import seed_everything
seed_everything(seed=123, workers=True)
hm_model = Classifier_hate_explain()
gpus = 1 if torch.cuda.is_available() else 0
trainer = pl.Trainer(gpus=gpus,max_epochs=60,callbacks=all_callbacks)
trainer.fit(hm_model, data_module)
Classifier_hate_model1 = Classifier_hate_explain.load_from_checkpoint('XXX/meme_text-explain30-val_f1_all_offn0.48.ckpt')
Classifier_hate_model1.to(device)
test_dataloader = DataLoader(dataset=te_p, batch_size=1478)
ckpt_path = 'XXX/meme_text-explain30-val_f1_all_offn0.48.ckpt' 
trainer.test(dataloaders=test_dataloader,ckpt_path=ckpt_path)
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
class Classifier_hate_explain_caption(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.MFB = MFB(768,768,True,256,64,0.1)
    self.loss_fn_emotion=torch.nn.KLDivLoss(reduction='batchmean',log_target=True)
    self.encode_text = torch.nn.Linear(1280,64)
    self.x_flatten=torch.nn.Linear(99840,768)
    self.y_flatten=torch.nn.Linear(39168,768)
    self.flatten=torch.nn.Flatten()
    self.fin_v = CoralLayer(size_in=64, num_classes=4)
    self.fin_sarcasm = torch.nn.Linear(64,3)
    self.fin_e1 = torch.nn.Linear(64,2)
    self.fin_e2 = torch.nn.Linear(64,2)
    self.fin_e3 = torch.nn.Linear(64,2)
    self.fin_e4 = torch.nn.Linear(64,2)
    self.fin_e5 = torch.nn.Linear(64,2)
  def forward(self, x,y):
      x_,y_ = x,y
      x = x.float()
      y = y.float()
      x=self.flatten(x)
      y=self.flatten(y)
      x=self.x_flatten(x)
      y=self.y_flatten(y)
      z_ = self.MFB(torch.unsqueeze(y,axis=1),torch.unsqueeze(x,axis=1))
      z = z_
      c_v = self.fin_v(torch.squeeze(z,dim=1))
      c_e1 = self.fin_e1(torch.squeeze(z,dim=1))
      c_e2 = self.fin_e2(torch.squeeze(z,dim=1))
      c_e3 = self.fin_e3(torch.squeeze(z,dim=1))
      c_e4 = self.fin_e4(torch.squeeze(z,dim=1))
      c_e5 = self.fin_e5(torch.squeeze(z,dim=1))
      c_sarcasm = self.fin_sarcasm(torch.squeeze(z,dim=1))
      c_e1 = torch.log_softmax(c_e1, dim=1)
      c_sarcasm = torch.log_softmax(c_sarcasm, dim=1)
      c_e2 = torch.log_softmax(c_e2, dim=1)
      c_e3 = torch.log_softmax(c_e3, dim=1)
      c_e4 = torch.log_softmax(c_e4, dim=1)
      
      c_e5 = torch.log_softmax(c_e5, dim=1)
      return z,c_v,c_e1,c_e2,c_e3,c_e4,c_e5,c_sarcasm
  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)
  def training_step(self, train_batch, batch_idx):
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity, name= train_batch
      
      lab = train_batch[lab]
      cap = train_batch[cap]
      img = train_batch[img]
      val = train_batch[val]
      e1 = train_batch[e1]
      e2 = train_batch[e2]
      e3 = train_batch[e3]
      e4 = train_batch[e4]
      e5 = train_batch[e5]
      sarcasm = train_batch[sarcasm]
      z,logit_val,a,b,c,d,e,logit_sarcasm= self.forward(cap,img) 
      val_level = levels_from_labelbatch(val, num_classes=4).type_as(logit_val)
      loss3  = coral_loss(logit_val, val_level)
      loss4 = self.cross_entropy_loss(a, e1)
      loss5 = self.cross_entropy_loss(b, e2)
      loss6 = self.cross_entropy_loss(c, e3)
      loss7 = self.cross_entropy_loss(d, e4)
      loss8 = self.cross_entropy_loss(e, e5)
      loss_sarcasm = self.cross_entropy_loss(logit_sarcasm, sarcasm)
      
      
      loss=loss3+loss4+loss5+loss6+loss7+loss8+loss_sarcasm
      self.log('train_loss', loss)
      
      return loss
  def validation_step(self, val_batch, batch_idx):
      
      
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity, name= val_batch
      cap = val_batch[cap]
      img = val_batch[img]
      val = val_batch[val]
      
      e1 = val_batch[e1]
      e2 = val_batch[e2]
      e3 = val_batch[e3]
      e4 = val_batch[e4]
      e5 = val_batch[e5]
      sarcasm = val_batch[sarcasm]
      z,logit_val,a,b,c,d,e,logit_sarcasm = self.forward(cap,img)
      val_level = levels_from_labelbatch(val, num_classes=4).type_as(logit_val)
      
      
      
      
      loss = coral_loss(logit_val, val_level)
      
      
      tmp = proba_to_label(torch.sigmoid(logit_val)).detach().cpu().numpy()
      
      val = val.detach().cpu().numpy()
      self.log('val_acc', f1_score(val,tmp,average='macro'))
      
      self.log('val_loss', loss)
      tqdm_dict = {'val_acc': accuracy_score(val,tmp)}
      
      return {
                'progress_bar': tqdm_dict,
              
              
              'val_acc e1': accuracy_score(e1.detach().cpu().numpy(),np.argmax(a.detach().cpu().numpy(),axis=-1)),
      'val_acc e2': accuracy_score(e2.detach().cpu().numpy(),np.argmax(b.detach().cpu().numpy(),axis=-1)),
      'val_acc e3': accuracy_score(e3.detach().cpu().numpy(),np.argmax(c.detach().cpu().numpy(),axis=-1)),
      'val_acc e4': accuracy_score(e4.detach().cpu().numpy(),np.argmax(d.detach().cpu().numpy(),axis=-1)),
      'val_acc e5': accuracy_score(e5.detach().cpu().numpy(),np.argmax(e.detach().cpu().numpy(),axis=-1)),
      'val_acc valence': f1_score(val,tmp,average='macro'),
       'val_acc sarcasm': accuracy_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1)),
       'f1 sarcasm': f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro')
      }
      
  def validation_epoch_end(self, validation_step_outputs):
    outs = []
    outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14,outs16,outs17 = \
    [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    outs15 = []
    outs18 = []
    for out in validation_step_outputs:
      outs.append(out['progress_bar']['val_acc'])
      outs1.append(out['val_acc e1'])
      outs2.append(out['val_acc e2'])
      outs3.append(out['val_acc e3'])
      outs4.append(out['val_acc e4'])
      outs5.append(out['val_acc e5'])
      outs14.append(out['val_acc valence'])
      
      
      outs17.append(out['val_acc sarcasm'])
      outs18.append(out['f1 sarcasm'])
    self.log('val_acc_all_offn', sum(outs)/len(outs))
    
    self.log('val_acc_all e1', sum(outs1)/len(outs1))
    self.log('val_acc_all e2', sum(outs2)/len(outs2))
    self.log('val_acc_all e3', sum(outs3)/len(outs3))
    self.log('val_acc_all e4', sum(outs4)/len(outs4))
    self.log('val_acc_all e5', sum(outs5)/len(outs5))
    self.log('val_acc_all valence', sum(outs14)/len(outs14))
    
    self.log('val_acc_all sarcasm', sum(outs17)/len(outs17))
    self.log('val_f1_all sarcasm', sum(outs18)/len(outs18))
    
    print(f'***valence f1 at epoch end {sum(outs)/len(outs)}****')
    print(f'***val acc valence at epoch end {sum(outs14)/len(outs14)}****')
    print(f'***val_acc_all e1 at epoch end {sum(outs1)/len(outs1)}****')
    print(f'***val_acc_all e2 at epoch end {sum(outs2)/len(outs2)}****')
    print(f'***val_acc_all e3 at epoch end {sum(outs3)/len(outs3)}****')
    print(f'***val_acc_all e4 at epoch end {sum(outs4)/len(outs4)}****')
    print(f'***val_acc_all e5 at epoch end {sum(outs5)/len(outs5)}****')
    print(f'***val acc sarcasm at epoch end {sum(outs17)/len(outs17)}****')
    print(f'***val f1 sarcasm at epoch end {sum(outs18)/len(outs18)}****')
  def test_step(self, batch, batch_idx):
      
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity, name= batch
      val = batch[val]
      
      cap = batch[cap]
      img = batch[img]
      e1 = batch[e1]
      e2 = batch[e2]
      e3 = batch[e3]
      e4 = batch[e4]
      e5 = batch[e5]
      sarcasm = batch[sarcasm]
      z,logit_val,a,b,c,d,e,logit_sarcasm = self.forward(cap,img)
      
      val_level = levels_from_labelbatch(val, num_classes=4).type_as(logit_val)
      
      
      
      
      loss = coral_loss(logit_val, val_level)
      tmp = proba_to_label(torch.sigmoid(logit_val)).detach().cpu().numpy()
      loss = coral_loss(logit_val, val_level)
      
      val = val.detach().cpu().numpy()
      self.log('test_acc', accuracy_score(val,tmp))
      self.log('test f1',f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro'))
      print(f'confusion matrix sarcasm {confusion_matrix(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1))}')
      print(f'confusion matrix valence {confusion_matrix(val,tmp)}')
      
      self.log('test_loss', loss)
      
      return {'test_loss': loss,
              'test_acc': accuracy_score(val,tmp),
              'test_acc e1': accuracy_score(e1.detach().cpu().numpy(),np.argmax(a.detach().cpu().numpy(),axis=-1)),
              'test_acc e2': accuracy_score(e2.detach().cpu().numpy(),np.argmax(b.detach().cpu().numpy(),axis=-1)),
              'test_acc e3': accuracy_score(e3.detach().cpu().numpy(),np.argmax(c.detach().cpu().numpy(),axis=-1)),
              'test_acc e4': accuracy_score(e4.detach().cpu().numpy(),np.argmax(d.detach().cpu().numpy(),axis=-1)),
              'test_acc e5': accuracy_score(e5.detach().cpu().numpy(),np.argmax(e.detach().cpu().numpy(),axis=-1)),
              'test_acc sarcasm': accuracy_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1)),
              'f1 sarcasm': f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro')}
  def test_epoch_end(self, outputs):
        
        outs = []
        outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14 = \
        [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        outs15 = []
        outs16 = []
        outs17 = []
        outs18 = []
        for out in outputs:
          
          outs.append(out['test_acc'])
          outs1.append(out['test_acc e1'])
          outs2.append(out['test_acc e2'])
          outs3.append(out['test_acc e3'])
          outs4.append(out['test_acc e4'])
          outs5.append(out['test_acc e5'])
          outs16.append(out['test_acc sarcasm'])
          
          outs18.append(out['f1 sarcasm'])
        
        self.log('final test f1', sum(outs)/len(outs))
        self.log('test_acc_all e1', sum(outs1)/len(outs1))
        self.log('test_acc_all e2', sum(outs2)/len(outs2))
        self.log('test_acc_all e3', sum(outs3)/len(outs3))
        self.log('test_acc_all e4', sum(outs4)/len(outs4))
        self.log('test_acc_all sarcasm', sum(outs16)/len(outs16))
        
        self.log('test_f1_all sarcasm', sum(outs18)/len(outs18))
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
    return optimizer

class HmDataModule(pl.LightningDataModule):
  def setup(self, stage):
    
      
    
    self.hm_train = t_p
    self.hm_val = v_p
    self.hm_test = te_p
    
  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=64)
  def val_dataloader(self):
    return DataLoader(self.hm_val, batch_size=64)
  def test_dataloader(self):
    return DataLoader(self.hm_test, batch_size=128)

data_module = HmDataModule()

checkpoint_callback = ModelCheckpoint(
     monitor='val_acc_all_offn',
     dirpath='ckpts_roman_caption_data/',
     filename='roman_caption_explain{epoch:02d}-val_f1_all_offn{val_acc_all_offn:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="max",
 )
all_callbacks = []
all_callbacks.append(checkpoint_callback)
from pytorch_lightning import seed_everything

seed_everything(seed=123, workers=True)
hm_model = Classifier_hate_explain_caption()
gpus = 1 if torch.cuda.is_available() else 0

trainer = pl.Trainer(gpus=gpus,max_epochs=60,callbacks=all_callbacks)
trainer.fit(hm_model, data_module)
Classifier_hate_model1_caption = Classifier_hate_explain_caption.load_from_checkpoint('XXX/M3P/roman_caption_explain45-val_f1_all_offn0.85.ckpt')
Classifier_hate_model1_caption.to(device)
test_dataloader = DataLoader(dataset=te_p, batch_size=1478)
ckpt_path = 'XXX/M3P/roman_caption_explain45-val_f1_all_offn0.85.ckpt' 
trainer.test(dataloaders=test_dataloader,ckpt_path=ckpt_path)

a = torch.randn(4,)
print(a)

b = torch.softmax(a, dim=-1)

class ContrastiveLoss(torch.nn.Module):
  def __init__(self, m=2.0):
    super(ContrastiveLoss, self).__init__()  
    self.m = m  
  def forward(self, y1, y2, d=0):
    
    
    
    euc_dist = torch.nn.functional.pairwise_distance(y1, y2)
    if d == 0:
      return torch.mean(torch.pow(euc_dist, 2))  
    else:  
      delta = self.m - euc_dist  
      delta = torch.clamp(delta, min=0.0, max=None)
      return torch.mean(torch.pow(delta, 2))  

def main():
  print("\nBegin contrastive loss demo \n")
  loss_func = ContrastiveLoss()
  y1 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).to(device)
  y2 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).to(device)
  y3 = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32).to(device)
  loss = loss_func(y1, y2, 0)
  print(loss)  
  loss = loss_func(y1, y2, 1)   
  print(loss)  
  loss = loss_func(y1, y3, 0)  
  print(loss)  
  loss = loss_func(y1, y3, 1)  
  print(loss)  
  print("\nEnd demo ")

if __name__ == "__main__":
  main()

import math

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

summed = 900 + 15000 + 800
weight = torch.tensor([900, 15000, 800]) / summed
print(weight)

pred_e = 0
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

class Classifier_final(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.MFB = MFB(768,768,True,256,64,0.1)
    self.loss_fn_emotion=torch.nn.KLDivLoss(reduction='batchmean',log_target=True)
    self.encode_text = torch.nn.Linear(1280,64)
    self.fin = torch.nn.Linear(64,2)
    self.fin_explain = torch.nn.Linear(64,2)
    self.fin_inten_explain = torch.nn.Linear(64,3)
    
    self.fin_caption = torch.nn.Linear(64,2)
    self.fin_inten_caption = torch.nn.Linear(64,3)
    
    self.fin_inten = torch.nn.Linear(64,3)
    
    self.loss_func = ContrastiveLoss()
    self.x_flatten=torch.nn.Linear(99840,768)
    self.y_flatten=torch.nn.Linear(39168,768)
    self.flatten=torch.nn.Flatten()
    self.mcb1 = torch.nn.Bilinear(64, 64, 64, bias=True)
    self.mcb2 = torch.nn.Bilinear(64, 64, 64, bias=True)
    self.mask=torch.tensor([0,1]).cuda()
    
    
  def forward(self, x,y,e,cap, off_label):
      
      x = x.float()
      y = y.float()
      x=self.flatten(x)
      y=self.flatten(y)
      x=self.x_flatten(x)
      y=self.y_flatten(y)
      z_ = self.MFB(torch.unsqueeze(y,axis=1),torch.unsqueeze(x,axis=1))
      z = z_
      values_ex_cap, attention_cap = scaled_dot_product(e, cap, z)  
      values_e=self.mcb1(z,e)
      values_cap=self.mcb2(z,cap)
      c = self.fin(torch.squeeze(values_ex_cap,dim=1))
      c_ex = self.fin_explain(torch.squeeze(values_e,dim=1))
      c_cap = self.fin_caption(torch.squeeze(values_cap,dim=1))
      c = torch.softmax(c, dim=-1)
      c_ex = torch.softmax(c_ex, dim=-1)
      c_cap= torch.softmax(c_cap, dim=-1)
      for i in range(len(off_label)):
        if (off_label[i]==1):
          c_inten = self.fin_inten(torch.squeeze(values_ex_cap,dim=1))
          c_inten_ex = self.fin_inten_explain(torch.squeeze(values_e,dim=1))
          c_inten_cap = self.fin_inten_caption(torch.squeeze(values_cap,dim=1))
        else: 
          
          c_inten = self.mask[0]*(self.fin_inten(torch.squeeze(values_ex_cap,dim=1)))
          c_inten_ex = self.mask[0]*(self.fin_inten_explain(torch.squeeze(values_e,dim=1)))
          c_inten_cap = self.mask[0]*(self.fin_inten_caption(torch.squeeze(values_cap,dim=1)))
      c_inten_cap = torch.softmax(c_inten_cap, dim=-1)
      c_inten_ex = torch.softmax(c_inten_ex, dim=-1)
      c_inten = torch.softmax(c_inten, dim=-1)
      return z,c,c_inten,c_ex,c_inten_ex,c_cap,c_inten_cap
  def cross_entropy_loss(self, logits, labels):
    
    weight_2=torch.tensor([1.0, 1.88, 2.09]).cuda()
    return F.nll_loss(logits, labels, weight=weight_2)
  def training_step(self, train_batch, batch_idx):
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity, name= train_batch
      lab = train_batch[lab]
      txt = train_batch[txt]
      cap = train_batch[cap]
      img = train_batch[img]
      e1 = train_batch[e1]
      e2 = train_batch[e2]
      e3 = train_batch[e3]
      e4 = train_batch[e4]
      e5 = train_batch[e5]
      intensity = train_batch[intensity]
      
      with torch.no_grad():
        z_ex,c_v,c_e1,c_e2,c_e3,c_e4,c_e5,c_sarcasm= Classifier_hate_model1(txt.to(device),img.to(device))
      with torch.no_grad():
        z_ex_cap,c_exp_v,c_exp_e1,c_exp_e2,c_exp_e3,c_exp_e4,c_exp_e5,c_exp_sarcasm= Classifier_hate_model1_caption(cap.to(device),img.to(device))
      z,logit_offen,inten,logit_offen_ex,inten_ex,logit_offen_cap,inten_cap= self.forward(txt,img,z_ex,z_ex_cap,lab) 
      loss17=self.cross_entropy_loss(inten, intensity)   
      loss11 = self.loss_func(logit_offen_cap, logit_offen,0)
      loss1=F.cross_entropy(logit_offen, lab)
      loss12 = self.loss_func(logit_offen_ex, logit_offen,0)
      loss1_explain= loss11+loss12
      loss1_explain_only=loss1*loss1_explain
      loss1_explain_final=loss1+loss1_explain_only
      loss171 = self.loss_func(inten, inten_cap,0)
      loss172 = self.loss_func(inten_ex, inten,0)
      loss17_explain= loss171+loss172
      loss17_explain_only=loss17*loss17_explain
      loss17_explain_final=loss17+loss17_explain_only
      tmp = np.argmax(logit_offen.detach().cpu().numpy(),axis=-1)
      loss=loss1_explain_final+loss17_explain_final
      self.log('train_loss', loss)
      return loss
  def validation_step(self, val_batch, batch_idx):
      
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity, name= val_batch
      
      lab = val_batch[lab]
      txt = val_batch[txt]
      img = val_batch[img]
      cap = val_batch[cap]
      intensity = val_batch[intensity]
      weight_2=torch.tensor([1.0, 1.88, 2.09])
      with torch.no_grad():
        z_ex,c_v,c_e1,c_e2,c_e3,c_e4,c_e5,c_sarcasm= Classifier_hate_model1(txt.to(device),img.to(device))
      with torch.no_grad():
        z_ex_cap,c_exp_v,c_exp_e1,c_exp_e2,c_exp_e3,c_exp_e4,c_exp_e5,c_exp_sarcasm= Classifier_hate_model1_caption(cap.to(device),img.to(device))
      _,logits,inten,logits_ex,inten_ex,logits_cap,inten_cap = self.forward(txt,img,z_ex,z_ex_cap,lab)
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss2=F.cross_entropy(logits, lab)
      loss1= self.loss_func(logits_cap, logits,0)
      loss3= self.loss_func(logits, logits_ex,0)
      loss_explain= loss1+loss3
      loss_explain_final=loss_explain*loss2 
      loss_off=loss2+loss_explain_final
      loss2_int=self.cross_entropy_loss(inten, intensity) 
      loss1_int= self.loss_func(inten, inten_ex,0)
      loss3_int= self.loss_func(inten, inten_cap,0)
      loss_explain_int= loss1_int+loss3_int
      loss_explain_final_int=loss_explain_int*loss2_int 
      loss_int=loss2_int+loss_explain_final_int
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss= loss_off+loss_int
      lab = lab.detach().cpu().numpy()
      self.log('val_acc', accuracy_score(lab,tmp))
      
      self.log('val_loss', loss)
      tqdm_dict = {'val_acc': accuracy_score(lab,tmp)}
      return {
                'progress_bar': tqdm_dict,
      'val_loss':loss,
      'val_f1 offensive': f1_score(lab,tmp,average='macro'),
        'val_acc intensity': accuracy_score(intensity.detach().cpu().numpy(), np.argmax(inten.detach().cpu().numpy(),axis=-1)),
       'f1 intensity': f1_score(intensity.detach().cpu().numpy(), np.argmax(inten.detach().cpu().numpy(),axis=-1),average='macro')
      }
      
  def validation_epoch_end(self, validation_step_outputs):
    outs = []
    outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14,outs16,outs17 = \
    [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    outs15 = []
    outs18 = []
    for out in validation_step_outputs:
      
      outs1.append(out['val_loss'])
      outs.append(out['progress_bar']['val_acc'])
      outs14.append(out['val_f1 offensive'])
      outs17.append(out['val_acc intensity'])
      outs18.append(out['f1 intensity'])
    
    self.log('val_loss', sum(outs1)/len(outs1))
    self.log('val_acc', sum(outs)/len(outs))
    self.log('val_f1 offensive', sum(outs14)/len(outs14))
    self.log('val_acc intensity', sum(outs17)/len(outs17))
    self.log('f1 intensity', sum(outs18)/len(outs18))
    print(f'***val_acc at epoch end {sum(outs)/len(outs)}****')
    
    print(f'***val_loss at epoch end {sum(outs1)/len(outs1)}****')
    print(f'***val_f1 offensive at epoch end {sum(outs14)/len(outs14)}****')
    print(f'***val_acc intensity at epoch end {sum(outs17)/len(outs17)}****')
    print(f'***val f1 intensity at epoch end {sum(outs18)/len(outs18)}****')
  def test_step(self, batch, batch_idx):
      
      txt,img,cap,lab,arou,val,e1,e2,e3,e4,e5,sarcasm,hum,intensity, name= batch
      lab = batch[lab]
      txt = batch[txt]
      cap = batch[cap]
      img = batch[img]
      intensity = batch[intensity]
      name= batch[name]
      weight_2=torch.tensor([1.0, 1.88, 2.09])
      with torch.no_grad():
        z_ex,c_v,c_e1,c_e2,c_e3,c_e4,c_e5,c_sarcasm= Classifier_hate_model1(txt.to(device),img.to(device))
      with torch.no_grad():
        z_ex_cap,c_exp_v,c_exp_e1,c_exp_e2,c_exp_e3,c_exp_e4,c_exp_e5,c_exp_sarcasm= Classifier_hate_model1_caption(cap.to(device),img.to(device))
      _,logits,inten,logits_ex,inten_ex,logits_cap,inten_cap, = self.forward(txt,img,z_ex,z_ex_cap,lab)
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss17  = F.cross_entropy(inten, intensity)
      loss17  = self.cross_entropy_loss(inten, intensity)
      loss1=F.cross_entropy(logits, lab)
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss= loss1+loss17
      lab = lab.detach().cpu().numpy()
      tmp_int = proba_to_label(torch.sigmoid(inten)).detach().cpu().numpy()
      self.log('test_acc', accuracy_score(lab,tmp))
      self.log('test f1',f1_score(lab,tmp,average='macro'))
      np.save('actual_label_offensive.npy',lab)
      np.save('predicted_offensive_after_explain.npy',tmp) 
      np.save('actual_label_intensity_after_explain.npy',intensity.detach().cpu().numpy())
      np.save('predicted_label_intensity_after_explain.npy',np.argmax(inten.detach().cpu().numpy(),axis=-1)) 
      np.save('name.npy',name)
      print(f'confusion matrix intensity {confusion_matrix(intensity.detach().cpu().numpy(), np.argmax(inten.detach().cpu().numpy(),axis=-1))}')
      print(f'confusion matrix offensive {confusion_matrix(lab,tmp)}')
      self.log('test_loss', loss)
      return {'test_loss': loss,
              'test_acc off': accuracy_score(lab,tmp),
              'test_f1 off': f1_score(lab,tmp,average='macro'),
              'test_acc intensity': accuracy_score(intensity.detach().cpu().numpy(), np.argmax(inten.detach().cpu().numpy(),axis=-1)),
              'f1 intensity': f1_score(intensity.detach().cpu().numpy(), np.argmax(inten.detach().cpu().numpy(),axis=-1),average='macro')}
  def test_epoch_end(self, outputs):
        
        outs = []
        outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14 = \
        [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        outs15 = []
        outs16 = []
        outs17 = []
        outs18 = []
        for out in outputs:
          
          outs.append(out['test_acc off'])
          outs16.append(out['test_acc intensity'])
          outs17.append(out['test_f1 off'])
          outs18.append(out['f1 intensity'])
        
        self.log('test_acc off', sum(outs)/len(outs))
        self.log('test_acc_all intensity', sum(outs16)/len(outs16))
        self.log('test_f1 off', sum(outs17)/len(outs17))
        self.log('test_f1_all intensity', sum(outs18)/len(outs18))
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
    return optimizer
class HmDataModule(pl.LightningDataModule):
  def setup(self, stage):
    self.hm_train = t_p
    self.hm_val = v_p
    self.hm_test = te_p
  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=64)
  def val_dataloader(self):
    return DataLoader(self.hm_val, batch_size=64)
  def test_dataloader(self):
    return DataLoader(self.hm_test, batch_size=128)

data_module = HmDataModule()
checkpoint_callback = ModelCheckpoint(
     monitor='f1 intensity',
    
     dirpath='ckpts_final_new/',
     filename='ckpts_final33-val_score_final_adding_multiplicative_loss{epoch:02d}-val_loss{f1 intensity:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="max",
 )
all_callbacks = []
all_callbacks.append(checkpoint_callback)
from pytorch_lightning import seed_everything

seed_everything(seed=1024, workers=True)

hm_model_final_hate_explain = Classifier_final()
gpus = 1 if torch.cuda.is_available() else 0

trainer = pl.Trainer(gpus=gpus,max_epochs=60,callbacks=all_callbacks)

trainer.fit(hm_model_final_hate_explain, data_module)
from pytorch_lightning import seed_everything

seed_everything(seed=123, workers=True)
hm_model_final_hate_explain = Classifier_final()
gpus = 1 if torch.cuda.is_available() else 0

trainer = pl.Trainer(gpus=gpus,max_epochs=60,callbacks=all_callbacks)

trainer.fit(hm_model_final_hate_explain, data_module)
test_dataloader = DataLoader(dataset=te_p, batch_size=1478)
ckpt_path_ex = 'XXX/ckpts_final/ckpts_final33-val_score_final_adding_multiplicative_loss0.69.ckpt' #correct code_loss value
trainer.test(dataloaders=test_dataloader,ckpt_path=ckpt_path_ex)
