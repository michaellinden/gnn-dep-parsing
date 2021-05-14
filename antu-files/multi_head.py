from .single import Attention
from ..modules import dy_model

import dynet as dy
import numpy as np



import math


@dy_model
class Attention:
    """
    Compute 'Scaled Dot Product Attention
    """

    def __call__(self, query, key, value, mask=None, dropout=None):

        tensor_mask = dy.inputTensor(mask, batched=True)
   
        sentence_length = query.dim()[0][0]
        batch_size = query.dim()[1]
        h = query.dim()[0][2]
        remaining_dim = query.dim()[0][1]


        query = dy.reshape(query, (sentence_length, remaining_dim * h), batch_size=batch_size)
        value = dy.reshape(value, (sentence_length, remaining_dim * h), batch_size=batch_size)
        key_transpose = dy.reshape(key, (remaining_dim * h, sentence_length), batch_size=batch_size)


        scores = (query * key_transpose) / math.sqrt(query.dim()[1])
        
        sentence_length = mask.shape[0]
        
        mask_arr = [ [1 for i in range(sentence_length)]  for j in range(sentence_length)]

        if mask is not None:
            for i in range(sentence_length):
                for j in range(sentence_length):
                    mask_arr[i][j] = np.multiply(mask[i], mask[j])

        mask_arr_np = np.array(mask_arr)
        mask_arr_tensor = dy.inputTensor(mask_arr_np, batched=True)

        scores = dy.cmult(scores, mask_arr_tensor)

        p_attn = dy.softmax(scores)

        return p_attn * value, p_attn


        # scores = torch.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size(-1))

        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)

        # p_attn = F.softmax(scores, dim=-1) # (5, 100)

        # if dropout is not None:
        #     p_attn = dropout(p_attn)

        # (5, 5) * ((5, 200), 100)
        # return torch.matmul(p_attn, value), p_attn





@dy_model
class MultiHeadedAttention:
    """
    Take in model size and number of heads.
    """

    def __init__(self, model, h, d_model, dropout=0.1):
        #super().__init__()
        assert d_model % h == 0

        pc = model.add_subcollection()

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = [pc.add_parameters((d_model, d_model)) for _ in range(3)]
        self.output_linear = pc.add_parameters((d_model, d_model))
        #self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        #self.output_linear = nn.Linear(d_model, d_model)
        
        self.attention = Attention()
        # self.attention = Attention()

        #self.dropout = nn.Dropout(p=dropout)

    def __call__(self, query, key, value, mask, drop):

        if type(query).__name__ == 'list':
            batch_size = query[0].dim()[1] #100

            sentence_length = len(query)

            np_query = np.array([q.npvalue() for q in query])
            query = dy.inputTensor(np_query, batched=True) # ((5, 200), 100)

        
        sentence_length = query.dim()[0][0]
        token_dim = query.dim()[0][1]
        batch_size = query.dim()[1]

        key = query
        value = query

        arr = []
        for l, x in zip(self.linear_layers, (query, key, value)):
            test = x * l
            
            curr_shape = test.dim()
            total_dim = curr_shape[0][0] * curr_shape[0][1] * curr_shape[1]

            test = dy.reshape(test, (total_dim // (batch_size * self.h * self.d_k), self.h, self.d_k), batch_size=batch_size) 

            test = dy.transpose(test, [0, 2, 1])

            arr.append(test)


        query, key, value = arr[0], arr[1], arr[2]
        

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=drop)

        res = x * self.output_linear
        print(res.dim())
        return res




        # 1) Do all the linear projections in batch from d_model => h x d_k
        #query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        #                     for l, x in zip(self.linear_layers, (query, key, value))]


        # 2) Apply attention on all the projected vectors in batch.
        #x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        #x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        #return self.output_linear(x)


class MultiLayerMultiHeadAttention:
    def __init__(self, model, h, d_model, num_layers=1, dropout=0.1):
        self.num_layers = num_layers
        self.layers = [MultiHeadedAttention(model, h, d_model) for _ in range(self.num_layers)]
        # final layer has to be dimension 800
        
        pc = model.add_subcollection()
        self.output_linear = pc.add_parameters((d_model, 800))

        # Save Variable
        self.pc = pc
        self.spec = (h, d_model, num_layers)
    

    def __call__(self, query, key, value, mask, drop):
        for i in range(self.num_layers):
            curr_layer = self.layers[i]
            query = curr_layer(query, key, value, mask, drop)

            key = query
            value = query
        
        return query * self.output_linear
    
    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        
        #cfg, vocabulary = spec
        #return GraphNNDecoder(model, cfg, vocabulary)
        
        h, d_model, num_layers = spec
        
        #token_repre, encoder, decoder = spec

        return MultiLayerMultiHeadAttention(model, h, d_model, num_layers=num_layers)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc