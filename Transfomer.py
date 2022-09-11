# This code is TRANSFORMER model by torch
from helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import math

__author__ = 'Wenter'

__all__ = ['TransformerDataset',
           'Embedding',
           'PositionalEncoding',
           'Attention',
           'MultiHeadedAttention',
           'Encoder',
           'Decoder',
           'Transformer']


"""
:论文地址:https://arxiv.org/abs/1706.03762v5
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Similar to the LSTM
class TransformerDataset(Dataset):
    def __init__(self,
                 data,
                 pad: int = 4794,
                 time_interval: int = 0):
        self.data = data
        self.date_index = self.data.index.get_level_values(0).unique()
        self.pad = pad
        self.time_interval = time_interval

    def __len__(self):
        return len(self.date_index)

    def __getitem__(self, idx):
        idx = self.date_index[idx]
        data = torch.tensor(self.data.loc[idx].values, dtype=torch.float, device=device)
        if data.shape[0] < self.pad:
            data = F.pad(data, (0, 0, self.pad - data.shape[0], 0), "constant", 0)

        features = data[:, :-1]

        if self.time_interval:

            response = data[self.time_interval - 1:, -1:]
        else:
            response = data[:, -1:]

        return features, response


class Embedding(nn.Module):
    """
    :class: input embedding
    """

    def __init__(self,
                 in_dim: int,
                 emb_dim: int = 20):
        """
        :param in_dim:input
        :param emb_dim: embedding size
        """
        super(Embedding, self).__init__()
        ##Like word embedding, using Linear to embedding features
        self.embed = nn.Linear(in_dim, emb_dim)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):
    """
    :class: as attention can't
    """
    def __init__(self, embedding=512, dropout=0.1, max_len=5000):
        """
        :func: init
        :param embedding: all dim
        :param dropout: dropout rate
        :param max_len: max time series length
        """
        # d_modl features
        super(PositionalEncoding, self).__init__()
        # dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize position matrix
        pe = torch.zeros(max_len, embedding).to(device)

        # position
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)

        # use the formula in the essay to give the position value
        # every position add every position mark
        div_term = torch.exp(torch.arange(0, embedding, 2).float() * (-math.log(10000.0) / embedding)).to(device)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register to get can save and load
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class Attention(nn.Module):
    """
    :class: attention model
    queryke
    """
    def __init__(self, embedding_dim: int = 512, mask: bool = None):
        """
        :func: query key value use Linear to input
        :param embedding_dim: input dim
        :param mask: whether see the future
        """
        super(Attention, self).__init__()

        self.q = nn.Linear(embedding_dim, embedding_dim)
        self.k = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Linear(embedding_dim, embedding_dim)

        # mask bool
        self.mask = mask

    def forward(self, q, k, v):
        """
        :func: attention process
               get the weight and get weight sum
               mask for the time series not to see the future
        :param q: query
        :param k: key
        :param v: value

        """
        # word length
        d_k = q.shape[-1]

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        # get the weight
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d_k)

        # mask
        if self.mask:
            # up triangle
            scores = torch.tril(scores)
            # for softmax
            scores = scores.masked_fill(scores == 0, -1e9)

        # softmax weight
        p_attn = F.softmax(scores, dim=-1)

        # get the value and weight
        return torch.matmul(p_attn, v), p_attn


class MultiHeadedAttention(nn.Module):
    """
    :class: Multi head attention
            n head attention combine
    """
    def __init__(self, head, embedding_dim=512, mask: bool = None):
        """
        :func: init
               each head has average dim
        :param head: head number
        :param embedding_dim: input dim
        :param mask: whether need mask attention(see the future)
        # Modulelist is list of nn model
        """
        super(MultiHeadedAttention, self).__init__()

        self.d_k = embedding_dim // head
        self.head = head

        self.multihead = nn.ModuleList(Attention(self.d_k, mask) for _ in range(head))

        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v):
        """
        :func: multiattention process
               get attention output concat
        :param q: query
        :param k: key
        :param v: value
        # idx_start,  idx_end _ different dim for multi head
        """
        out = torch.tensor([]).to(device)
        for i, a in enumerate(self.multihead):
            idx_start = i * self.d_k
            idx_end = (i + 1) * self.d_k
            o, _ = a(q[:, :, idx_start:idx_end], k[:, :, idx_start:idx_end], v[:, :, idx_start:idx_end])

            out = torch.cat((out, o), 2)

        out = self.linear(out)
        return out


class Encoder(nn.Module):
    """
    :class: Encoder combine needed model
    """
    def __init__(self, head, embedding_dim, length, hidden_dim, dropout, mask: bool = None):
            """
            :func: init
            :param head: head number
            :param embedding_dim: input
            :param length: time length
            :param hidden_dim: fc hidden dim
            :param dropout: dropout rate
            :param mask: whether mask (see future)
            # add layer norm same in the essay
            """
            super(Encoder, self).__init__()

            self.multiattention = MultiHeadedAttention(head, embedding_dim, mask)
            self.dropout1 = nn.Dropout(dropout)

            self.norm1 = nn.LayerNorm((length, embedding_dim))

            self.linear1 = nn.Linear(embedding_dim, hidden_dim)

            self.linear2 = nn.Linear(hidden_dim, embedding_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm((length, embedding_dim))

    def forward(self, x):
            """
            :func: encoder process
                   same in the essay
            """
            out = self.multiattention(x, x, x)
            out = self.norm1(x + out)
            out = self.dropout1(out)
            x = out

            out = self.linear2(F.elu(self.linear1(out)))
            out = self.norm2(x + out)
            out = self.dropout1(out)

            return out


class Decoder(nn.Module):
    """
    :class: Decoder combine needed model
    """
    def __init__(self, head, embedding_dim, length, hidden_dim, dropout, mask):
        """
        :func: init
        :param head: head number
        :param embedding_dim: input
        :param length: time length
        :param hidden_dim: fc hidden dim
        :param dropout: dropout rate
        :param mask: whether mask (see future)
        # add layer norm same in the essay
        """
        super(Decoder, self).__init__()

        self.maskmultihead1 = MultiHeadedAttention(head, embedding_dim, mask)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm((length, embedding_dim))

        self.maskmultihead2 = MultiHeadedAttention(head, embedding_dim, mask)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm((length, embedding_dim))

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm((length, embedding_dim))

    def forward(self, x, kv):
        """
        :func: encoder process
               same in the essay
        # in maskmultihead2, kv use the encoder output
        """
        out = self.maskmultihead1(x, x, x)
        out = self.norm1(x + out)
        out = self.dropout1(out)
        x = out

        out = self.maskmultihead2(out, kv, kv)
        out = self.norm2(x + out)
        out = self.dropout2(out)
        x = out

        out = self.linear2(F.elu(self.linear1(out)))
        out = self.norm3(x + out)
        out = self.dropout3(out)

        return out


class Transformer(nn.Module):
    """
    :class: Transformer(combine encoder and decoder)
    """
    def __init__(self,
                 n:int,
                 head:int,
                 input_dim:int,
                 embedding_dim:int,
                 output_dim:int,
                 length:int,
                 hidden_dim:int,
                 dropout:int):
        """
        :func: init
        :param n: coder number
        :param head: head number
        :param input_dim: feature dim
        :param embedding_dim:.
        :param output_dim: response dim
        :param length: time length
        :param hidden_dim: fc hidden dim
        :param dropout: dropout rate
        """

        super(Transformer, self).__init__()

        # input
        self.embedding1 = Embedding(input_dim,embedding_dim)
        self.position = PositionalEncoding(embedding_dim, dropout)

        # encoder
        self.encoder = nn.ModuleList(
            Encoder(head, embedding_dim, length, hidden_dim, dropout, mask=True) for _ in range(n))

        # decoder
        self.embedding2 = Embedding(output_dim,embedding_dim)
        self.decoder = nn.ModuleList(
            Decoder(head, embedding_dim, length, hidden_dim, dropout, mask=True) for _ in range(n))

        # output
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        """
        :func: Transformer process,
               iter encoder get the output of kv from features,
               output need  to be shifted as this time index can't see next,
               iter the decoder get the output and fc it
        :param x: input features
        :param y: output response shifted before
        :return: our fit value
        """
        # input
        x = self.embedding1(x)
        x = self.position(x)

        # encoder
        for _, e in enumerate(self.encoder):
            x = e(x)

        # output
        # need be shifted before
        y = self.embedding2(y)
        y = self.position(y)

        # decoder
        for _, d in enumerate(self.decoder):
            y = d(y, x)

        # output
        y = self.dropout(y)
        output = self.linear1(y)
        output = F.elu(output)
        output = self.linear2(output)

        return output


# same as the LSTM process
class Transfomer_Process:
    DROP_OUT: int = 0.2
    BATCH_SIZE: int = 1
    EPOCHS: int = 10
    ITER: int = 32
    LR: int = 0.001
    CONVERGE: int = 0.001
    SAVE: list = []

    def __init__(self,):
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.time_interval = 0

    @staticmethod
    def flog(epoch, content):
        print(f"------epoch{epoch}----{content}------")

    def initialize(self,
                   new: bool = True,
                   all_dates: list = ALL_DATES,
                   input_dim: int = 1,
                   hidden_dim: int = 20,
                   layer_dim: int = 1,
                   output_dim: int = 1,
                   time_interval: int = 0

    ):

        '''
        dp = DataProcessing(all_dates,validate=0.2,train=0.6
        train,val,test = dp.data_for_nn()
        save_gzip(train, './ML/lstm/lstm_train_data.pkl')
        save_gzip(val,'./ML/lstm/lstm_val_data.pkl')
        save_gzip(test,'./ML/lstm/lstm_test_data.pkl')

        '''
        train = load_gzip('./ML/transformer/transformer_train_data.pkl')
        val = load_gzip('./ML/transformer/transformer_val_data.pkl')
        test = load_gzip('./ML/transformer/transformer_test_data.pkl')

        self.time_interval = time_interval
        if not new:
            try:
                self.model = torch.load(f"./ML/transformer/transformer_model_{self.time_interval}_train.pth")
            except ValueError as e:
                print('----ERROR! no existing model----')

        else:
            self.model = Transformer(1,2,13,20,1,4794,100,self.DROP_OUT)

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

        train_dataset = TransformerDataset(train)
        val_dataset = TransformerDataset(val)
        test_dataset = TransformerDataset(test)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.BATCH_SIZE)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.BATCH_SIZE)

        self.flog(0,f"Transformer_time{self.time_interval}_model initialize successfully")
        self.flog(0,"Transformer")

        sum_ = 0
        for name, param in self.model.named_parameters():
            mul = 1
            for size_ in param.shape:
                mul *= size_  # 统计每层参数个数
            sum_ += mul  # 累加每层参数个数
            print(f"{name}:{param.shape}")
        print('参数个数：', sum_)

    def _base_line(self):
        loss = []
        for _,val in enumerate(self.val_loader):
                response = val[1]
                val_loss = self.criterion(response[:,:-1,:],response[:,1:,:])
                loss.append(val_loss)
        loss = torch.tensor(loss, dtype=torch.float)
        self.flog(0,f"the base line loss is {torch.mean(loss)}")
        return torch.mean(loss)

    def train(self):
        self.model.train()
        base_line_loss = self._base_line()
        validate_loss = self._validate(0)
        self.flog(0,"train beginning")

        for i in range(self.EPOCHS):
            sum_train_loss = []
            for _, train in enumerate(self.train_loader):
                feature = train[0]
                response = train[1]
                response_shift = torch.cat((torch.zeros(response.shape[0],self.time_interval,1).to(device),response[:,:-5,:]),1)

                for _ in range(self.ITER):
                    self.optimizer.zero_grad()
                    output = self.model(feature,response_shift)

                    loss = self.criterion(output,response)
                    loss.backward()

                    self.optimizer.step()

                sum_train_loss.append(loss)
            sum_train_loss = torch.tensor(sum_train_loss, dtype=torch.float)
            '''if abs(sum_train_loss-epoch_loss)/epoch_loss <= self.CONVERGE:
                print(f"----Epoch{i} Loss Converge----")
                break

            epoch_loss = sum_train_loss'''
            self.flog(i+1,f"the train loss is {torch.mean(sum_train_loss)}")

            if (val_l := self._validate(i+1)) <= base_line_loss:
                if val_l < validate_loss:
                    self.flog(i+1,"have improvement")
                    self.flog(i + 1, "save model parameters")

                    torch.save(self.model, f"./ML/transformer/transformer_model_{self.time_interval}_train.pth")
                    validate_loss = val_l
            else:
                self.flog(i+1,"model is not good")

            if np.isnan(val_l):
                self.model = torch.load(f"./ML/transformer/transformer_model_{self.time_interval}_train.pth")
                validate_loss = self._validate(i+1)
                self.flog(i+1,"the model nan retrain")



            '''if (i+1) % 5 == 0:
                print(f"----save model parameters ----")
                torch.save(self.model,'./ML/lstm/lstm_model_train.pth')'''



        self.flog('final', f"the final loss is {torch.mean(sum_train_loss)}")

    def _validate(self,epoch):
        loss = []
        with torch.no_grad():
            for _, val in enumerate(self.val_loader):
                feature = val[0]
                response = val[1]
                response_shift = torch.cat((torch.zeros(response.shape[0], self.time_interval, 1).to(device), response[:, :-5, :]),
                                           1)

                output = self.model(feature,response_shift)
                loss.append(self.criterion(output, response))
            loss = torch.tensor(loss, dtype=torch.float)
            self.flog(epoch,f"the val loss is {torch.mean(loss)}")

        return torch.mean(loss)

    def test(self):
        loss = []
        sign = []
        with torch.no_grad():
            for _, test in enumerate(self.test_loader):
                feature = test[0]
                response = test[1]
                response_shift = torch.cat((torch.zeros(response.shape[0],self.time_interval,1).to(device),response[:,:-5,:]),1)

                output = self.model(feature,response_shift)

                loss.append(self.criterion(output, response))

                direction_right = ((torch.sign(output) == torch.sign(response)).sum())/(response.shape[0]*response.shape[1])
                sign.append(direction_right)
            loss = torch.tensor(loss,dtype=torch.float)
            sign = torch.tensor(sign,dtype=torch.float)

            self.flog("test", f"the test loss is {torch.mean(loss)}")
            self.flog("test", f"the test sign accurate is {torch.mean(sign)}")

    def adjust(self):
        lr = [0.1,0.01,0.001,0.001]
        pass



if __name__ == "__main__":
    transformer = Transfomer_Process()
    transformer.initialize(new=True,input_dim=13,hidden_dim=100,layer_dim=2,output_dim=1,  time_interval=5)
    transformer.train()
    transformer.test()



