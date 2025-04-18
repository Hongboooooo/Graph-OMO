import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.bool), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1) # b x ls x ls
    
    return subsequent_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head # 4
        self.d_model = d_model # 512
        self.d_k = d_k # 256
        self.d_v = d_v # 256

        self.w_q = nn.Linear(d_model, n_head*d_k)
        self.w_k = nn.Linear(d_model, n_head*d_k)
        self.w_v = nn.Linear(d_model, n_head*d_v)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0/(d_model+d_v)))

        self.temperature = np.power(d_k, 0.5)
        self.attn_dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(n_head*d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, q, k, v, mask=None):
        # q: BS X T X D, k: BS X T X D, v: BS X T X D, mask: BS X T X T 
        bs, n_q, _ = q.shape
        bs, n_k, _ = k.shape
        bs, n_v, _ = v.shape

        assert n_k == n_v

        residual = q

        q = self.w_q(q).view(bs, n_q, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_q, self.d_k)
        k = self.w_k(k).view(bs, n_k, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, self.d_k)
        v = self.w_v(v).view(bs, n_v, self.n_head, self.d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, self.d_v)
        # (bs, nf, d) -> (bs, nf, n_h*d_k) -> (bs, nf, n_h, d_k) -> (n_h, bs, nf, d_k) -> (n_h*bs, nf, d_k)
        # (bs, nf, d) -> (bs, nf, n_h*d_v) -> (bs, nf, n_h, d_v) -> (n_h, bs, nf, d_v) -> (n_h*bs, nf, d_v)
        # .contiguous() makes sure the tensor is stored continuously in the memory, which guarantees .view() won't report error

        attn = torch.bmm(q, k.transpose(1, 2)) # attn: (n_h*bs, nf, nf) | compute correlation of features at each frame of the sequence
        attn = attn / self.temperature

        if mask is not None: # mask is None by default
            mask = mask.repeat(self.n_head, 1, 1) # (n_head*bs) x n_q x n_k 
            attn = attn.masked_fill(mask, -np.inf) # where is true, where is -inf

        attn = F.softmax(attn, dim=2) # attn: (n_h*bs, nf, nf) 
        
        attn = self.attn_dropout(attn)
        output = torch.bmm(attn, v) #  attn: (n_h*bs, nf, d_v) | update feature at each frame according to the correlations between that feature and other features at different frames

        output = output.view(self.n_head, bs, n_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, n_q, -1)
        # BS X n_q X (n_head*dv)

        # output = self.fc(output) # BS X n_q X D
        output = self.dropout(self.fc(output)) # BS X n_q X D
        output = self.layer_norm(output + residual) # BS X n_q X D

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid): # d_in 512 d_hid 512
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: BS X N X D
        residual = x
        output = x.transpose(1, 2) # BS X D X N
        output = self.w_2(F.relu(self.w_1(output))) # BS X D X N
        output = output.transpose(1, 2) # BS X N X D
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # BS X N X D

        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model)

    def forward(self, decoder_input, self_attn_time_mask, self_attn_padding_mask): # padding_mask is used here
        # decode_input: BS X T X D
        # time_mask: BS X T X T (padding postion are ones)
        # padding_mask: BS X T (padding position are zeros, diff usage from above)
        bs, dec_len, dec_hidden = decoder_input.shape
        
        decoder_out, dec_self_attn = self.self_attn(decoder_input, decoder_input, decoder_input, \
                                mask=self_attn_time_mask)
        # BS X T X D, BS X T X T
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()
        # BS X T X D

        # Explain:
        # assume: 
        #   decoder_out: (BS, T, D)
        #   self_attn_padding_mask: (BS, T)
        # then:
        #   self_attn_padding_mask.unsqueeze(-1): (BS, T, 1)
        #   self_attn_padding_mask.unsqueeze(-1).float(): True, False | 1.0, 0.0
        #   decoder_out *= self_attn_padding_mask.unsqueeze(-1).float():
        #       have: 
        #           decoder_out | (2, 4, 3) = 
        #           [
        #             [
        #               [1,2,1],
        #               [2,3,4],
        #               [6,3,5],
        #               [1,2,3]
        #             ],
        #             [
        #               [3,2,5],
        #               [4,5,3],
        #               [1,3,5],
        #               [4,3,2]
        #             ]
        #           ]
        #           self_attn_padding_mask.unsqueeze(-1).float(): | (2, 4, 1) = 
        #           [
        #             [
        #               [1.0],
        #               [1.0],
        #               [0.0],
        #               [0.0]
        #             ],
        #             [
        #               [1.0],
        #               [1.0],
        #               [1.0],
        #               [0.0]
        #             ]
        #           ]
        #       broadcasting along last dimension:
        #           self_attn_padding_mask.unsqueeze(-1).float(): | (2, 4, 3) = 
        #           [
        #             [
        #               [1.0, 1.0, 1.0],
        #               [1.0, 1.0, 1.0],
        #               [0.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0]
        #             ],
        #             [
        #               [1.0, 1.0, 1.0],
        #               [1.0, 1.0, 1.0],
        #               [1.0, 1.0, 1.0],
        #               [0.0, 0.0, 0.0]
        #             ]
        #           ]
        #       multiplying:
        #           decoder_out *= self_attn_padding_mask.unsqueeze(-1).float(): | (2, 4, 3) = 
        #           [
        #             [
        #               [1.0, 2.0, 1.0],
        #               [2.0, 3.0, 4.0],
        #               [0.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0]
        #             ],
        #             [
        #               [3.0, 2.0, 5.0],
        #               [4.0, 5.0, 3.0],
        #               [1.0, 3.0, 5.0],
        #               [0.0, 0.0, 0.0]
        #             ]
        #           ]

        decoder_out = self.pos_ffn(decoder_out) # BS X T X D
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float() # zero out the padding part at the back of feeded sequences 

        return decoder_out, dec_self_attn
        # BS X T X D, BS X T X T


class Decoder(nn.Module):
    def __init__(
            self,
            d_feats, # 2x3 + 3 + 256
            d_model, # 512
            n_layers, # 4
            n_head, # 4
            d_k,  # 256
            d_v,  # 256
            max_timesteps,  # 120+1
            use_full_attention=False):
        super(Decoder, self).__init__()

        self.start_conv = nn.Conv1d(d_feats, d_model, 1) # (input: 2x3 + 3 + 256)
        self.position_vec = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_timesteps+1, d_model, padding_idx=0),
            freeze=True) # (bs, num_idx) -> (bs, num_idx, dim_vec)
            # embedding layer takes an index as input and return the embedded vector of the given index
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])

        self.use_full_attention = use_full_attention 

    def forward(self, decoder_input, padding_mask, decoder_pos_vec, obj_embedding=None):
        # decoder_input: BS X D X T 
        # padding_mask: BS X 1 X T | 
        # decoder_pos_vec: BS X 1 X T | (bs, 1, 121) [1,2,3,...,121]
        # obj_embedding: BS X 1 X D | (bs, 1, 512)

        dec_self_attn_list = []

        padding_mask = padding_mask.squeeze(1) # BS X T
        decoder_pos_vec = decoder_pos_vec.squeeze(1) # BS X T

        input_embedding = self.start_conv(decoder_input)  # BS X D X T
        input_embedding = input_embedding.transpose(1, 2) # BS X T X D
        if obj_embedding is not None: # default is not none | actually is the diffusion time step embedding
            new_input_embedding = torch.cat((obj_embedding, input_embedding), dim=1) # BS X (1+T) X D 
        else:
            new_input_embedding = input_embedding

        # self.position_vec = self.position_vec.cuda()
        pos_embedding = self.position_vec(decoder_pos_vec) # BS X T X D | bs x 121 x 512
        
        # Time mask is same for all blocks, while padding mask differ according to the position of block
        if self.use_full_attention: # use_full_attention is True by default
            time_mask = None
        else:
            time_mask = get_subsequent_mask(decoder_pos_vec) 
        # BS X T X T (Prev steps are 0, later 1)
       
        dec_output = new_input_embedding + pos_embedding # BS X T X D | BS x nf x 512
            # make feature of each frame has information about the position in the sequence 
        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn = dec_layer(
                dec_output, # BS X T X D
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T | padding_mask only makes difference when seq_len < 120

            dec_self_attn_list += [dec_self_attn]

        return dec_output, dec_self_attn_list
        # BS X T X D, list
