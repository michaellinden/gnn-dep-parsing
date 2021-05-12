from ..modules import dy_model

import dynet as dy
import numpy as np

import math


@dy_model
class ScaledDotProductAttention:
    def __init__(self, d_model, attention_dropout=0.1):
        #super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        # TODO: self.dropout = nn.Dropout(attention_dropout)
        #self.softmax = nn.Softmax(dim=-1)

    def __call__(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat] or (batch * d_l) x max_len x d_k
        # k: [batch, slot, feat] or (batch * d_l) x max_len x d_k
        # v: [batch, slot, feat] or (batch * d_l) x max_len x d_v
        # q in LAL is (batch * d_l) x 1 x d_k

        dim = q.dim()

        q = dy.reshape(q, (dim[1], dim[2]), batch_size=dim[0])
        k = dy.reshape(k, (dim[1], dim[2]), batch_size=dim[0])
        v = dy.reshape(v, (dim[1], dim[2]), batch_size=dim[0])

        k_tranpose = dy.transpose(k, [1, 0, 2])

        attn = (q * k_tranpose) / self.temper


        sentence_length = attn_mask.shape[0]
        
        mask_arr = [ [1 for i in range(sentence_length)]  for j in range(sentence_length)]

        if attn_mask is not None:
            for i in range(sentence_length):
                for j in range(sentence_length):
                    mask_arr[i][j] = np.multiply(attn_mask[i], attn_mask[j])

        mask_arr_np = np.array(mask_arr)
        mask_arr_tensor = dy.inputTensor(mask_arr_np, batched=True)

        attn = dy.cmult(attn, mask_arr_tensor)

        p_attn = dy.softmax(attn)

        output = p_attn * v

        return output, p_attn



        # attn = torch.bmm(q, k.transpose(1, 2)) / self.temper # (batch * d_l) x max_len x max_len
        # # in LAL, gives: (batch * d_l) x 1 x max_len
        # # attention weights from each word to each word, for each label
        # # in best model (repeated q): attention weights from label (as vector weights) to each word

        # if attn_mask is not None:
        #     assert attn_mask.size() == attn.size(), \
        #             'Attention mask shape {} mismatch ' \
        #             'with Attention logit tensor shape ' \
        #             '{}.'.format(attn_mask.size(), attn.size())

        #     attn.data.masked_fill_(attn_mask, -float('inf'))

        # attn = self.softmax(attn)
        # # Note that this makes the distribution not sum to 1. At some point it
        # # may be worth researching whether this is the right way to apply
        # # dropout to the attention.
        # # Note that the t2t code also applies dropout in this manner
        # attn = self.dropout(attn)
        # output = torch.bmm(attn, v) # (batch * d_l) x max_len x d_v
        # # in LAL, gives: (batch * d_l) x 1 x d_v

        # return output, attn