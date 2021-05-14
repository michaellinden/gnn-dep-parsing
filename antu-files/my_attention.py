from ..modules import dy_model

import dynet as dy
import numpy as np

from ..modules.linear import Linear
from ..modules.feed_forward import PositionwiseFeedForward

import math


@dy_model
class ScaledDotProductAttention:
    def __init__(self, model, d_model, attention_dropout=0.1):
        #super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        
        pc = model.add_subcollection()
        self.pc = pc

        self.spec = d_model
        # TODO: self.dropout = nn.Dropout(attention_dropout)
        #self.softmax = nn.Softmax(dim=-1)
    
    def convert_list_to_tensor(self, q):
        batch_size = q[0].dim()[1] #100

        sentence_length = len(q)

        np_query = np.array([a.npvalue() for a in q])
        q = dy.inputTensor(np_query, batched=True) # ((5, 200), 100)

        return q

    def __call__(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat] or (batch * d_l) x max_len x d_k
        # k: [batch, slot, feat] or (batch * d_l) x max_len x d_k
        # v: [batch, slot, feat] or (batch * d_l) x max_len x d_v
        # q in LAL is (batch * d_l) x 1 x d_k

        if type(q).__name__ == 'list':
            q = self.convert_list_to_tensor(q)
            k = self.convert_list_to_tensor(k)
            v = self.convert_list_to_tensor(v)
            

        dim = q.dim()

        #q = dy.reshape(q, (dim[1], dim[2]), batch_size=dim[0])
        #k = dy.reshape(k, (dim[1], dim[2]), batch_size=dim[0])
        #v = dy.reshape(v, (dim[1], dim[2]), batch_size=dim[0])

        k_tranpose = dy.transpose(k, [1, 0])

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

        return output



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
    
    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        
        #cfg, vocabulary = spec
        #return GraphNNDecoder(model, cfg, vocabulary)
        
        d_model = spec

        return ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        
        #token_repre, encoder, decoder = spec

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc


# class LayerNormalization(nn.Module):
#     def __init__(self, d_hid, eps=1e-3, affine=True):
#         super(LayerNormalization, self).__init__()

#         self.eps = eps
#         self.affine = affine
#         if self.affine:
#             self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
#             self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

#     def forward(self, z):
#         if z.size(-1) == 1:
#             return z

#         mu = torch.mean(z, keepdim=True, dim=-1)
#         sigma = torch.std(z, keepdim=True, dim=-1)
#         ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
#         if self.affine:
#             ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

#         return ln_out



@dy_model
class MyMultiHeadAttention:
    """
    Multi-head attention module
    """

    def __init__(self, hparams, n_head, d_model, d_k, d_v, residual_dropout=0.1, attention_dropout=0.1, d_positional=None, model=None):
        
        pc = model.add_subcollection()
        
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.hparams = hparams

        self.partitioned = False

        # if d_positional is None:
        #     self.partitioned = False
        # else:
        #     self.partitioned = True

        self.w_qs = pc.add_parameters((d_model, d_k), init='normal')
        self.w_ks = pc.add_parameters((d_model, d_k), init='normal')
        self.w_vs = pc.add_parameters((d_model, d_v), init='normal')

        # if self.partitioned:
        #     self.d_content = d_model - d_positional
        #     self.d_positional = d_positional

        #     self.w_qs1 = pc.add_parameters((n_head, self.d_content, d_k // 2), init='normal')
        #     self.w_ks1 = pc.add_parameters((n_head, self.d_content, d_k // 2), init='normal')
        #     self.w_vs1 = pc.add_parameters((n_head, self.d_content, d_v // 2), init='normal')

        #     self.w_qs2 = pc.add_parameters((n_head, self.d_positional, d_k // 2), init='normal')
        #     self.w_ks2 = pc.add_parameters((n_head, self.d_positional, d_k // 2), init='normal')
        #     self.w_vs2 = pc.add_parameters((n_head, self.d_positional, d_v // 2), init='normal')

        #     #self.w_qs1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_k // 2))
        #     #self.w_ks1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_k // 2))
        #     #self.w_vs1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_v // 2))

        #     #self.w_qs2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_k // 2))
        #     #self.w_ks2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_k // 2))
        #     #self.w_vs2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_v // 2))

        #     # init.xavier_normal_(self.w_qs1)
        #     # init.xavier_normal_(self.w_ks1)
        #     # init.xavier_normal_(self.w_vs1)

        #     # init.xavier_normal_(self.w_qs2)
        #     # init.xavier_normal_(self.w_ks2)
        #     # init.xavier_normal_(self.w_vs2)
        # else:
        #     self.w_qs = pc.add_parameters((d_model, d_k), init='normal')
        #     self.w_ks = pc.add_parameters((d_model, d_k), init='normal')
        #     self.w_vs = pc.add_parameters((d_model, d_v), init='normal')

        #     #self.w_qs = pc.add_parameters((n_head, d_model, d_k), init='normal')
        #     #self.w_ks = pc.add_parameters((n_head, d_model, d_k), init='normal')
        #     #self.w_vs = pc.add_parameters((n_head, d_model, d_v), init='normal')

        #     #self.w_qs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
        #     #self.w_ks = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
        #     #self.w_vs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_v))

        #     #init.xavier_normal_(self.w_qs)
        #     #init.xavier_normal_(self.w_ks)
        #     #init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(model, d_model, attention_dropout=attention_dropout)

        #self.layer_norm = LayerNormalization(d_model)

        self.proj = Linear(pc, n_head*d_v, d_model, bias=False)

        # if not self.partitioned:
        #     # The lack of a bias term here is consistent with the t2t code, though
        #     # in my experiments I have never observed this making a difference.
        #     # len_inp x (n_head * d_v) -> len_inp, d_model
        #     self.proj = Linear(pc, n_head*d_v, d_model, bias=False)
        #     #self.proj = nn.Linear(n_head*d_v, d_model, bias=False)
        # else:
        #     self.proj1 = Linear(pc, n_head*(d_v//2), self.d_content, bias=False)
        #     self.proj2 = Linear(pc, n_head*(d_v//2), self.d_positional, bias=False)

            #self.proj1 = nn.Linear(n_head*(d_v//2), self.d_content, bias=False)
            #self.proj2 = nn.Linear(n_head*(d_v//2), self.d_positional, bias=False)
        
        self.output_linear = pc.add_parameters((d_model, 800))
        self.pc = pc
        
        self.spec = (hparams, n_head, d_model, d_k, d_v, residual_dropout, attention_dropout, d_positional)
        # TODO: self.residual_dropout = FeatureDropout(residual_dropout)

    def repeat(self, inp, num_repetitions):
        inp_dim = inp.dim()
        new_dim = (1, inp_dim[0][0], inp_dim[0][1])
        inp = dy.reshape(inp, new_dim, batch_size=inp_dim[1])
        v_inp_repeated = dy.concatenate(self.n_head * [inp])
        return v_inp_repeated


    def split_qkv_packed(self, inp, qk_inp=None):
        #print(inp.dim()) # ((sent_len, d_model), batch_size)

        # MARK: split_qkv_packed section
        #inp_dim = inp.dim()
        #new_dim = (1, inp_dim[0][0], inp_dim[0][1])
        #inp = dy.reshape(inp, new_dim, batch_size=inp_dim[1])
        #v_inp_repeated = dy.concatenate(self.n_head * [inp])
        #print(inp.dim()) # want: (n_head, sent_len, d_model, batch_size)



        v_inp = inp

        if qk_inp is None:
            qk_inp = v_inp
        # else:
        #     qk_inp_dim = qk_inp.dim()
        #     v_new_dim = (1, qk_inp[0][0], qk_inp[0][1])
        #     v_inp = dy.reshape(qk_inp, v_new_dim, batch_size=v_new_dim[1])
        #     v_inp_repeated = dy.concatenate(self.n_head * [v_inp])

        self.partitioned = False
        if not self.partitioned:
            #print(qk_inp.dim()) # ((n_head, sent_len, d_model), batch_size)
            # self.w_qs.dim() # (n_head, d_model, d_k)
            # (n_head * sent_len, B) * (B, d_k) 
            q_s_unrepeated = qk_inp * self.w_qs
            k_s_unrepeated = qk_inp * self.w_ks
            v_s_unrepeated = v_inp * self.w_vs

            q_s = self.repeat(q_s_unrepeated, self.n_head)
            k_s = self.repeat(k_s_unrepeated, self.n_head)
            v_s = self.repeat(v_s_unrepeated, self.n_head)
   


        else:
            pass # TODO
    
        
        return q_s, k_s, v_s

        # v_inp_repeated = inp.repeat(self.n_head, 1).view(self.n_head, -1, inp.size(-1)) # n_head x len_inp x d_model
        # if qk_inp is None:
        #     qk_inp_repeated = v_inp_repeated
        # else:
        #     qk_inp_repeated = qk_inp.repeat(self.n_head, 1).view(self.n_head, -1, qk_inp.size(-1))

        # if not self.partitioned:
        #     q_s = torch.bmm(qk_inp_repeated, self.w_qs) # n_head x len_inp x d_k
        #     k_s = torch.bmm(qk_inp_repeated, self.w_ks) # n_head x len_inp x d_k
        #     v_s = torch.bmm(v_inp_repeated, self.w_vs) # n_head x len_inp x d_v
        # else:
        #     q_s = torch.cat([
        #         torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_qs1),
        #         torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_qs2),
        #         ], -1)
        #     k_s = torch.cat([
        #         torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_ks1),
        #         torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_ks2),
        #         ], -1)
        #     v_s = torch.cat([
        #         torch.bmm(v_inp_repeated[:,:,:self.d_content], self.w_vs1),
        #         torch.bmm(v_inp_repeated[:,:,self.d_content:], self.w_vs2),
        #         ], -1)
        # return q_s, k_s, v_s

    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        pass


        
        # # Input is padded representation: n_head x len_inp x d
        # # Output is packed representation: (n_head * mb_size) x len_padded x d
        # # (along with masks for the attention and output)
        # n_head = self.n_head
        # d_k, d_v = self.d_k, self.d_v

        # len_padded = batch_idxs.max_len
        # mb_size = batch_idxs.batch_size
        # q_padded = q_s.new_zeros((n_head, mb_size, len_padded, d_k))
        # k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
        # v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
        # invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=DTYPE)

        # for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
        #     q_padded[:,i,:end-start,:] = q_s[:,start:end,:]
        #     k_padded[:,i,:end-start,:] = k_s[:,start:end,:]
        #     v_padded[:,i,:end-start,:] = v_s[:,start:end,:]
        #     invalid_mask[i, :end-start].fill_(False)

        # return(
        #     q_padded.view(-1, len_padded, d_k),
        #     k_padded.view(-1, len_padded, d_k),
        #     v_padded.view(-1, len_padded, d_v),
        #     invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1),
        #     (~invalid_mask).repeat(n_head, 1),
        #     )

    def combine_v(self, outputs):
        #print('about to combine v')
        #print(outputs.dim()) # ((4, 21, 32), 5) ((n_head, sentence_len, d_kv), batch_size

        self.partitioned = False
        if not self.partitioned:
            outputs = dy.transpose(outputs, [1, 0, 2])
            outputs_dim = outputs.dim()
            #print(outputs.dim())
        
            outputs = dy.reshape(outputs, (outputs_dim[0][0], outputs_dim[0][1] * outputs_dim[0][2]), batch_size=outputs_dim[1])

            outputs = self.proj(outputs)
        else:
            pass # TODO
        
        return outputs
        
        # # Combine attention information from the different heads
        # n_head = self.n_head
        # outputs = outputs.view(n_head, -1, self.d_v) # n_head x len_inp x d_kv

        # if not self.partitioned:
        #     # Switch from n_head x len_inp x d_v to len_inp x (n_head * d_v)
        #     outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, n_head * self.d_v)

        #     # Project back to residual size
        #     outputs = self.proj(outputs)
        # else:
        #     d_v1 = self.d_v // 2
        #     outputs1 = outputs[:,:,:d_v1]
        #     outputs2 = outputs[:,:,d_v1:]
        #     outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, n_head * d_v1)
        #     outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, n_head * d_v1)
        #     outputs = torch.cat([
        #         self.proj1(outputs1),
        #         self.proj2(outputs2),
        #         ], -1)

        # return outputs

    def __call__(self, inp, attn_mask, qk_inp=None):
        if type(inp).__name__ == 'list':
            inp = self.convert_list_to_tensor(inp)
        residual = inp


        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_padded, k_padded, v_padded = self.split_qkv_packed(inp, qk_inp=qk_inp)
        # n_head x len_inp x d_kv

        # Switch to padded representation, perform attention, then switch back
        #q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)
        # (n_head * batch) x len_padded x d_kv

        

        outputs = []
        for i in range(self.n_head):
            output = self.attention(
            q_padded[i], k_padded[i], v_padded[i],
            attn_mask=attn_mask
            )

            output_dim = output.dim()
            output = dy.reshape(output, (1, output_dim[0][0], output_dim[0][1]), batch_size=output_dim[1])

            outputs.append(output)


        outputs = dy.concatenate(outputs)

        outputs = self.combine_v(outputs)

        return outputs * self.output_linear

        # outputs_padded, attns_padded = self.attention(
        #     q_padded, k_padded, v_padded,
        #     attn_mask=attn_mask,
        #     )
        

        #outputs = outputs_padded[output_mask]
        # (n_head * len_inp) x d_kv
        #outputs = self.combine_v(outputs)
        # len_inp x d_model

        #outputs = self.residual_dropout(outputs, batch_idxs)

        #return self.layer_norm(outputs + residual), attns_padded


    
    def convert_list_to_tensor(self, q):
        batch_size = q[0].dim()[1] #100

        sentence_length = len(q)

        np_query = np.array([a.npvalue() for a in q])
        q = dy.inputTensor(np_query, batched=True) # ((5, 200), 100)

        return q
    
    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        
        #cfg, vocabulary = spec
        #return GraphNNDecoder(model, cfg, vocabulary)
        
        hparams, n_head, d_model, d_k, d_v, residual_dropout, attention_dropout, d_positional = spec
        
        #token_repre, encoder, decoder = spec

        return MultiLayerMultiHeadAttention(hparams, n_head, d_model, d_k, d_v, residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=d_positional, model=model)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc











class LabelAttention:
    """
    Single-head Attention layer for label-specific representations
    """

    def __init__(self, hparams, d_model, d_k, d_v, d_l, d_proj, use_resdrop=True, q_as_matrix=False, residual_dropout=0.1, attention_dropout=0.1, d_positional=None, model=None):
        pc = model.add_subcollection()

        self.hparams = hparams
        self.d_k = d_k
        self.d_v = d_v
        self.d_l = d_l # Number of Labels
        self.d_model = d_model # Model Dimensionality
        self.d_proj = d_proj # Projection dimension of each label output
        self.use_resdrop = use_resdrop # Using Residual Dropout?
        self.q_as_matrix = q_as_matrix # Using a Matrix of Q to be multiplied with input instead of learned q vectors
        #self.combine_as_self = hparams.lal_combine_as_self # Using the Combination Method of Self-Attention
        self.combine_as_self = True # ? 

        self.partitioned = False
        # if d_positional is None:
        #     self.partitioned = False
        # else:
        #     self.partitioned = True

        if self.q_as_matrix:
            self.w_qs = pc.add_parameters((self.d_l, d_model, d_k), init='normal')
            #self.w_qs = nn.Parameter(torch_t.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
        else:
            #self.w_qs = pc.add_parameters((self.d_l, d_k), init='normal')
            #self.w_ks = pc.add_parameters((self.d_l, d_model, d_k), init='normal')
            #self.w_vs = pc.add_parameters((self.d_l, d_model, d_v), init='normal')

            self.w_qs = pc.add_parameters((1, d_k), init='normal')
            self.w_ks = pc.add_parameters((d_model, d_k), init='normal')
            self.w_vs = pc.add_parameters((d_model, d_v), init='normal')

            # self.w_qs = nn.Parameter(torch_t.FloatTensor(self.d_l, d_k), requires_grad=True)
            # self.w_ks = nn.Parameter(torch_t.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
            # self.w_vs = nn.Parameter(torch_t.FloatTensor(self.d_l, d_model, d_v), requires_grad=True)

            #init.xavier_normal_(self.w_qs)
            #init.xavier_normal_(self.w_ks)
            #init.xavier_normal_(self.w_vs)

        # if self.partitioned:
        #     self.d_content = d_model - d_positional
        #     self.d_positional = d_positional

        #     if self.q_as_matrix:
        #         self.w_qs1 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_content, d_k // 2), requires_grad=True)
        #     else:
        #         self.w_qs1 = nn.Parameter(torch_t.FloatTensor(self.d_l, d_k // 2), requires_grad=True)
        #     self.w_ks1 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_content, d_k // 2), requires_grad=True)
        #     self.w_vs1 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_content, d_v // 2), requires_grad=True)

        #     if self.q_as_matrix:
        #         self.w_qs2 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_positional, d_k // 2), requires_grad=True)
        #     else:
        #         self.w_qs2 = nn.Parameter(torch_t.FloatTensor(self.d_l, d_k // 2), requires_grad=True)
        #     self.w_ks2 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_positional, d_k // 2), requires_grad=True)
        #     self.w_vs2 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_positional, d_v // 2), requires_grad=True)

        #     init.xavier_normal_(self.w_qs1)
        #     init.xavier_normal_(self.w_ks1)
        #     init.xavier_normal_(self.w_vs1)

        #     init.xavier_normal_(self.w_qs2)
        #     init.xavier_normal_(self.w_ks2)
        #     init.xavier_normal_(self.w_vs2)
        # else:
        #     if self.q_as_matrix:
        #         self.w_qs = nn.Parameter(torch_t.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
        #     else:
        #         self.w_qs = nn.Parameter(torch_t.FloatTensor(self.d_l, d_k), requires_grad=True)
        #     self.w_ks = nn.Parameter(torch_t.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
        #     self.w_vs = nn.Parameter(torch_t.FloatTensor(self.d_l, d_model, d_v), requires_grad=True)

        #     init.xavier_normal_(self.w_qs)
        #     init.xavier_normal_(self.w_ks)
        #     init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(model, d_model, attention_dropout=attention_dropout)

        #self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        
        # if self.combine_as_self:
        #     self.layer_norm = LayerNormalization(d_model)
        # else:
        #     self.layer_norm = LayerNormalization(self.d_proj)

        if not self.partitioned:
            # The lack of a bias term here is consistent with the t2t code, though
            # in my experiments I have never observed this making a difference.
            if self.combine_as_self:
                self.proj = Linear(pc, self.d_l * d_v, d_model, bias=False)
            else:
                self.proj = Linear(pc, d_v, d_model, bias=False) # input dimension does not match, should be d_l * d_v
        else:
            if self.combine_as_self:
                self.proj1 = Linear(pc, self.d_l*(d_v//2), self.d_content, bias=False)
                self.proj2 = Linear(pc, self.d_l*(d_v//2), self.d_positional, bias=False)
            else:
                self.proj1 = Linear(pc, d_v//2, self.d_content, bias=False)
                self.proj2 = Linear(pc, d_v//2, self.d_positional, bias=False)
        if not self.combine_as_self:
            self.reduce_proj = Linear(pc, d_model, self.d_proj, bias=False)

        self.output_linear = pc.add_parameters((d_model, 800))
        self.pc = pc
        self.spec = (hparams, d_model, d_k, d_v, d_l, d_proj, use_resdrop, q_as_matrix, residual_dropout, attention_dropout, d_positional)
        #self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, k_inp=None):
        v_inp = inp

        if k_inp is None:
            k_inp  = v_inp
        
        self.partitioned = False

        if not self.partitioned:
            if self.q_as_matrix:
                q_s = k_inp * self.w_qs
                # print(q_s.dim())
                # print('NOT MATRIX')
                # exit(0)
            else:
                q_s = self.w_qs
                #q_s_dim = q_s.dim() # ((112, 128), 1) # (d_l, d_k)
                #print(q_s_dim)
                #q_s = dy.reshape(q_s, (q_s_dim[0][0], 1, q_s_dim[0][1]))
                #q_s = dy.reshape(q_s, (1, q_s_dim[0]))

            
   
            k_s = k_inp * self.w_ks
            v_s = v_inp * self.w_vs

            q_s = self.repeat(q_s, self.d_l)
            k_s = self.repeat(k_s, self.d_l)
            v_s = self.repeat(v_s, self.d_l)
            
        else:
            pass # TODO
        
        return q_s, k_s, v_s
        
        # len_inp = inp.size(0)
        # v_inp_repeated = inp.repeat(self.d_l, 1).view(self.d_l, -1, inp.size(-1)) # d_l x len_inp x d_model
        # if k_inp is None:
        #     k_inp_repeated = v_inp_repeated
        # else:
        #     k_inp_repeated = k_inp.repeat(self.d_l, 1).view(self.d_l, -1, k_inp.size(-1)) # d_l x len_inp x d_model

        # if not self.partitioned:
        #     if self.q_as_matrix:
        #         q_s = torch.bmm(k_inp_repeated, self.w_qs) # d_l x len_inp x d_k
        #     else:
        #         q_s = self.w_qs.unsqueeze(1) # d_l x 1 x d_k
        #     k_s = torch.bmm(k_inp_repeated, self.w_ks) # d_l x len_inp x d_k (21, 128)
        #     v_s = torch.bmm(v_inp_repeated, self.w_vs) # d_l x len_inp x d_v
        # else:
        #     if self.q_as_matrix:
        #         q_s = torch.cat([
        #             torch.bmm(k_inp_repeated[:,:,:self.d_content], self.w_qs1),
        #             torch.bmm(k_inp_repeated[:,:,self.d_content:], self.w_qs2),
        #             ], -1)
        #     else:
        #         q_s = torch.cat([
        #             self.w_qs1.unsqueeze(1),
        #             self.w_qs2.unsqueeze(1),
        #             ], -1)
        #     k_s = torch.cat([
        #         torch.bmm(k_inp_repeated[:,:,:self.d_content], self.w_ks1),
        #         torch.bmm(k_inp_repeated[:,:,self.d_content:], self.w_ks2),
        #         ], -1)
        #     v_s = torch.cat([
        #         torch.bmm(v_inp_repeated[:,:,:self.d_content], self.w_vs1),
        #         torch.bmm(v_inp_repeated[:,:,self.d_content:], self.w_vs2),
        #         ], -1)
        # return q_s, k_s, v_s
    
    def repeat(self, inp, num_repetitions):
        inp_dim = inp.dim()
        new_dim = (1, inp_dim[0][0], inp_dim[0][1])
        inp = dy.reshape(inp, new_dim, batch_size=inp_dim[1])
        v_inp_repeated = dy.concatenate(num_repetitions * [inp])
        return v_inp_repeated
    
    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        pass
        # Input is padded representation: n_head x len_inp x d
        # Output is packed representation: (n_head * mb_size) x len_padded x d
        # (along with masks for the attention and output)
        # n_head = self.d_l
        # d_k, d_v = self.d_k, self.d_v

        # len_padded = batch_idxs.max_len
        # mb_size = batch_idxs.batch_size
        # if self.q_as_matrix:
        #     q_padded = q_s.new_zeros((n_head, mb_size, len_padded, d_k))
        # else:
        #     q_padded = q_s.repeat(mb_size, 1, 1) # (d_l * mb_size) x 1 x d_k
        # k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
        # v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
        # invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=DTYPE)

        # for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
        #     if self.q_as_matrix:
        #         q_padded[:,i,:end-start,:] = q_s[:,start:end,:]
        #     k_padded[:,i,:end-start,:] = k_s[:,start:end,:]
        #     v_padded[:,i,:end-start,:] = v_s[:,start:end,:]
        #     invalid_mask[i, :end-start].fill_(False)

        # if self.q_as_matrix:
        #     q_padded = q_padded.view(-1, len_padded, d_k)
        #     attn_mask = invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1)
        # else:
        #     attn_mask = invalid_mask.unsqueeze(1).repeat(n_head, 1, 1)
        
        # output_mask = (~invalid_mask).repeat(n_head, 1)

        # return(
        #     q_padded,
        #     k_padded.view(-1, len_padded, d_k),
        #     v_padded.view(-1, len_padded, d_v),
        #     attn_mask,
        #     output_mask,
        #     )

    def combine_v(self, outputs):
        self.partitioned = False

        if not self.partitioned:
            outputs = dy.transpose(outputs, [1, 0, 2])
            outputs_dim = outputs.dim()

            outputs = dy.reshape(outputs, (outputs_dim[0][0], outputs_dim[0][1] * outputs_dim[0][2]), batch_size=outputs_dim[1])
            outputs = self.proj(outputs)
        
        return outputs

        # Combine attention information from the different labels
        # d_l = self.d_l
        # outputs = outputs.view(d_l, -1, self.d_v) # d_l x len_inp x d_v

        # if not self.partitioned:
        #     # Switch from d_l x len_inp x d_v to len_inp x d_l x d_v
        #     if self.combine_as_self:
        #         outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, d_l * self.d_v)
        #     else:
        #         outputs = torch.transpose(outputs, 0, 1)#.contiguous() #.view(-1, d_l * self.d_v)
        #     # Project back to residual size
        #     outputs = self.proj(outputs) # Becomes len_inp x d_l x d_model
        # else:
        #     d_v1 = self.d_v // 2
        #     outputs1 = outputs[:,:,:d_v1]
        #     outputs2 = outputs[:,:,d_v1:]
        #     if self.combine_as_self:
        #         outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, d_l * d_v1)
        #         outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, d_l * d_v1)
        #     else:
        #         outputs1 = torch.transpose(outputs1, 0, 1)#.contiguous() #.view(-1, d_l * d_v1)
        #         outputs2 = torch.transpose(outputs2, 0, 1)#.contiguous() #.view(-1, d_l * d_v1)
        #     outputs = torch.cat([
        #         self.proj1(outputs1),
        #         self.proj2(outputs2),
        #         ], -1)#.contiguous()

        # return outputs

    def convert_list_to_tensor(self, q):
        batch_size = q[0].dim()[1] #100

        sentence_length = len(q)

        np_query = np.array([a.npvalue() for a in q])
        q = dy.inputTensor(np_query, batched=True) # ((5, 200), 100)

        return q

    def __call__(self, inp, attn_mask, k_inp=None):
        if type(inp).__name__ == 'list':
            inp = self.convert_list_to_tensor(inp)

        print('printing input dimension ')
        print(inp.dim())
        residual = inp

        inp_dim = inp.dim() # (sent_len, d_model), batch_size



        # len_inp = inp.size(0)

        # # While still using a packed representation, project to obtain the
        # # query/key/value for each head
        q_s, k_s, v_s = self.split_qkv_packed(inp, k_inp=k_inp)
        # # d_l x len_inp x d_k
        # # q_s is d_l x 1 x d_k

        outputs = []
        for i in range(self.d_l):
            output = self.attention(
            q_s[i], k_s[i], v_s[i],
            attn_mask=attn_mask
            )

            output_dim = output.dim()
            output = dy.reshape(output, (1, output_dim[0][0], output_dim[0][1]), batch_size=output_dim[1])

            outputs.append(output)
        
        outputs = dy.concatenate(outputs)
        
        outputs = self.combine_v(outputs)

        return outputs * self.output_linear

        # # Switch to padded representation, perform attention, then switch back
        # q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)
        # # q_padded, k_padded, v_padded: (d_l * batch_size) x max_len x d_kv
        # # q_s is (d_l * batch_size) x 1 x d_kv

        # outputs_padded, attns_padded = self.attention(
        #     q_padded, k_padded, v_padded,
        #     attn_mask=attn_mask,
        #     )
        # # outputs_padded: (d_l * batch_size) x max_len x d_kv
        # # in LAL: (d_l * batch_size) x 1 x d_kv
        # # on the best model, this is one value vector per label that is repeated max_len times
        # if not self.q_as_matrix:
        #     outputs_padded = outputs_padded.repeat(1,output_mask.size(-1),1)
        # outputs = outputs_padded[output_mask]
        # # outputs: (d_l * len_inp) x d_kv or LAL: (d_l * len_inp) x d_kv
        # # output_mask: (d_l * batch_size) x max_len
        # torch.cuda.empty_cache()

        # outputs = self.combine_v(outputs)
        # # outputs: len_inp x d_l x d_model, whereas a normal self-attention layer gets len_inp x d_model
        # if self.use_resdrop:
        #     if self.combine_as_self:
        #         outputs = self.residual_dropout(outputs, batch_idxs)
        #     else:
        #         outputs = torch.cat([self.residual_dropout(outputs[:,i,:], batch_idxs).unsqueeze(1) for i in range(self.d_l)], 1)
        # if self.combine_as_self:
        #     outputs = self.layer_norm(outputs + inp)
        # else:
        #     for l in range(self.d_l):
        #         outputs[:, l, :] = outputs[:, l, :] + inp
            
        #     outputs = self.reduce_proj(outputs) # len_inp x d_l x d_proj
        #     outputs = self.layer_norm(outputs) # len_inp x d_l x d_proj
        #     outputs = outputs.view(len_inp, -1).contiguous() # len_inp x (d_l * d_proj)
        
        # return outputs, attns_padded
    
    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        
        #cfg, vocabulary = spec
        #return GraphNNDecoder(model, cfg, vocabulary)
        
        hparams, d_model, d_k, d_v, d_l, d_proj, use_resdrop, q_as_matrix, residual_dropout, attention_dropout, d_positional = spec
        
        #token_repre, encoder, decoder = spec

        return LabelAttention(hparams, d_model, d_k, d_v, d_l, d_proj, use_resdrop=use_resdrop, q_as_matrix=q_as_matrix, residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=d_positional, model=model)

        #return MultiLayerMultiHeadAttention(hparams, n_head, d_model, d_k, d_v, residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=d_positional, model=model)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc





class Encoder:
    def __init__(self, hparams, d_model,
                    num_layers=1, num_heads=2, d_kv = 32, d_ff=1024, d_l=112,
                    d_positional=None,
                    num_layers_position_only=0,
                    relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1,
                    use_lal=True,
                    lal_d_kv=128,
                    lal_d_proj=128,
                    lal_resdrop=True,
                    lal_pwff=True,
                    lal_q_as_matrix=False,
                    lal_partitioned=True,
                    model=None):

        pc = model.add_subcollection()
        lal_combine_as_self = False # can change

        #self.embedding_container = [embedding]
        #d_model = embedding.d_embedding
        self.hparams = hparams

        d_k = d_v = d_kv

        self.stacks = []

        for i in range(num_layers):
            attn = MyMultiHeadAttention(hparams, num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout,
                                      attention_dropout=attention_dropout, d_positional=d_positional, model=pc)
            ff = PositionwiseFeedForward(pc, 800, 800)
            # if d_positional is None:
            #     ff = PositionwiseFeedForward(pc, d_model, d_ff)
            #     # ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout,
            #     #                              residual_dropout=residual_dropout)
            # else:
            #     pass # TODO
            #     # ff = PartitionedPositionwiseFeedForward(d_model, d_ff, d_positional, relu_dropout=relu_dropout,
            #     #                                         residual_dropout=residual_dropout)

            #self.add_module(f"attn_{i}", attn)
            #self.add_module(f"ff_{i}", ff)

            self.stacks.append((attn, ff))

        if use_lal:
            lal_d_positional = d_positional if lal_partitioned else None
            attn = LabelAttention(hparams, d_model, lal_d_kv, lal_d_kv, d_l, lal_d_proj, use_resdrop=lal_resdrop, q_as_matrix=lal_q_as_matrix,
                                  residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=lal_d_positional, model=pc)
            ff = PositionwiseFeedForward(pc, 800, 800)
            # ff_dim = lal_d_proj * d_l
            # # if lal_combine_as_self:
            # #     ff_dim = d_model
            # # if lal_pwff:
            # #     if d_positional is None or not lal_partitioned:
            # #         ff = PositionwiseFeedForward(pc, ff_dim, d_ff)
            # #         #ff = PositionwiseFeedForward(ff_dim, d_ff, relu_dropout=relu_dropout, residual_dropout=residual_dropout)
            # #     else:
            # #         pass
            # #         #ff = PartitionedPositionwiseFeedForward(ff_dim, d_ff, d_positional, relu_dropout=relu_dropout, residual_dropout=residual_dropout)
            # # else:
            # #     ff = None

            #self.add_module(f"attn_{num_layers}", attn)
            #self.add_module(f"ff_{num_layers}", ff)
            self.stacks.append((attn, ff))

        self.num_layers_position_only = num_layers_position_only
        # if self.num_layers_position_only > 0:
        #     assert d_positional is None, "num_layers_position_only and partitioned are incompatible"

        
        self.pc = pc
        self.spec = (hparams, d_model,
                    num_layers, num_heads, d_kv, d_ff, d_l,
                    d_positional,
                    num_layers_position_only,
                    relu_dropout, residual_dropout, attention_dropout,
                    use_lal,
                    lal_d_kv,
                    lal_d_proj,
                    lal_resdrop,
                    lal_pwff,
                    lal_q_as_matrix,
                    lal_partitioned)

    def __call__(self, xs,  batch_idxs):
        #emb = self.embedding_container[0]
        #res, res_c, timing_signal, batch_idxs = emb(xs, pre_words_idxs, batch_idxs, extra_content_annotations=extra_content_annotations)

        for i, (attn, ff) in enumerate(self.stacks):
            if type(xs) == list:
                print('before res has len {} and dim {}'.format(len(xs), xs[0].dim()))
            else:
                print('before res has dim {}'.format(xs.dim()))
            res = attn(xs, batch_idxs)
            print('res dimension is: ')
            print(res.dim())
            #ff = None
            if ff is not None:
                #print(res.dim())
                #print(ff.dim())
                res = ff(res)
                #print(res.dim())
                #res = ff(res, batch_idxs)


        return res #batch_idxs
    
    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        
        #cfg, vocabulary = spec
        #return GraphNNDecoder(model, cfg, vocabulary)

        hparams, d_model, num_layers, num_heads, d_kv, d_ff, d_l,d_positional,num_layers_position_only,relu_dropout, residual_dropout, attention_dropout,use_lal,lal_d_kv,lal_d_proj,lal_resdrop,lal_pwff,lal_q_as_matrix,lal_partitioned = spec
        
        #hparams, d_model, num_layers, num_heads, d_kv, d_ff, d_l, d_positional, num_layers_position_only, relu_dropout, residual_dropout, attention_dropout, use_lal,lal_d_kv,lal_d_proj,lal_resdrop,lal_pwff,lal_q_as_matrix,lal_partitioned = spec
        
        #token_repre, encoder, decoder = spec
        return Encoder(hparams, d_model, num_layers=num_layers, num_heads=num_heads, d_kv=d_kv, d_ff=d_ff, d_l=d_l, d_positional=d_positional, num_layers_position_only=num_layers_position_only, relu_dropout=relu_dropout, residual_dropout=residual_dropout, attention_dropout=attention_dropout, use_lal=use_lal, lal_d_kv=lal_d_kv, lal_d_proj=lal_d_proj, lal_resdrop=lal_resdrop, lal_pwff=lal_pwff, lal_q_as_matrix=lal_q_as_matrix,lal_partitioned=lal_partitioned, model=model)

        #return LabelAttention(hparams, d_model, d_k, d_v, d_l, d_proj, use_resdrop=use_resdrop, q_as_matrix=q_as_matrix, residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=d_positional, model=model)

        #return MultiLayerMultiHeadAttention(hparams, n_head, d_model, d_k, d_v, residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=d_positional, model=model)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc