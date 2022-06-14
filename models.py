import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
import random
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
from transformers import VisualBertModel
import config

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(TextEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout,
                            bidirectional=True)

    def forward(self, x, length):
        self.lstm.flatten_parameters()
        length = length.squeeze(1)
        x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(x)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output


class Attention(nn.Module):
    """Implements additive attention and return the attention vector used to weight the values.
        Additive attention consists in concatenating key and query and then passing them trough a linear layer."""

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)

    def forward(self, hidden, enc_output, obj_emb, length=None):
        # key = [batch size, dec hid dim]
        # queries = [batch size, src sent len, enc hid dim]
        hidden = hidden.unsqueeze(1)
        bs, max_len, _ = enc_output.size()
        enc_output = self.attn(enc_output)

        attention_v = torch.sum(hidden * enc_output, dim=2)

        attention_obj = torch.sum(obj_emb * enc_output, dim=2)

        ratio = 0.99
        attention = ratio * attention_v + (1 - ratio) * attention_obj

        if length is not None:
            padding_mask = torch.arange(0, max_len).type_as(length).unsqueeze(0).expand(bs, max_len)
            padding_mask = ~padding_mask.lt(length)
            attention.masked_fill_(padding_mask, -math.inf)

        return F.softmax(attention, dim=1)


class ObjectAttention(nn.Module):
    """
    Object Attention Network
    """
    def __init__(self, objects_dim, decoder_dim, attention_dim, dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(ObjectAttention, self).__init__()
        self.features_att = weight_norm(nn.Linear(objects_dim, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))   # linear layer to transform decoder's output
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        
    def forward(self, object_features, decoder_hidden):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # print(f'image_features:', image_features.shape)
        att1 = self.features_att(object_features)  # (batch_size, 36, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 36)
        attention_weighted_encoding = (object_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)

        return attention_weighted_encoding


class Decoder(nn.Module):
    def __init__(self, decoder_dim, embed_dim, vision_dim, vocab_size, dropout=0.3):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTMCell(embed_dim + vision_dim, decoder_dim, bias=True)  # top down attention LSTMCell
        self.obj_attetion = ObjectAttention(embed_dim, decoder_dim, decoder_dim)
        self.vision_attention = Attention(vision_dim, decoder_dim)
        self.out = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding_input, h1, c1, fusion_feature, obj_embedding):
        # obj_feature = self.obj_attetion(obj_embedding, h1)
        vision_score = self.vision_attention(h1, fusion_feature, obj_embedding)
        vision_weighted = torch.bmm(vision_score.unsqueeze(1), fusion_feature).squeeze(1)
        rnn_input = torch.cat((embedding_input, vision_weighted), dim=1)
        h1, c1 = self.rnn(rnn_input, (h1, c1))
        output = self.out(self.dropout(h1))
        return output, h1, c1


class MultiDecoder(nn.Module):
    def __init__(self, decoder_dim, embed_dim, vision_dim, vocab_size, dropout=0.3):
        super(MultiDecoder, self).__init__()
        self.rnn = nn.LSTMCell(embed_dim * 3 + vision_dim, decoder_dim, bias=True)  # top down attention LSTMCell

        self.vision_attention = Attention(vision_dim, decoder_dim)
        self.one_hot_attention = Attention(embed_dim * 2, decoder_dim)
        self.out = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding_input, h1, c1, fusion, one_hot_sentence, one_hot_sentence_len):
        vision_score = self.vision_attention(h1, fusion)
        vision_weighted = torch.bmm(vision_score.unsqueeze(1), fusion).squeeze(1)

        one_hot_score = self.one_hot_attention(h1, one_hot_sentence, one_hot_sentence_len)
        one_hot_weighted = torch.bmm(one_hot_score.unsqueeze(1), one_hot_sentence).squeeze(1)

        rnn_input = torch.cat((embedding_input, vision_weighted, one_hot_weighted), dim=1)

        h1, c1 = self.rnn(rnn_input, (h1, c1))
        output = self.out(self.dropout(h1))
        return output, h1, c1


class SinkhornNetwork(nn.Module):

    def __init__(self, N, n_iters, tau):
        """
        :param N: N degree vector. How many objects
        :param n_iters:  iteration times
        :param tau:
        """
        super(SinkhornNetwork, self).__init__()
        self.n_iters = n_iters
        self.tau = tau

        self.W1_txt = nn.Linear(512, 128)
        self.W1_vis = nn.Linear(2048, 512)
        self.W2_vis = nn.Linear(512, 128)
        self.W1_sen = nn.Linear(512, 128)  # 128 -> 256
        self.W_fc_pos = nn.Linear(388, 256)
        self.W_fc = nn.Linear(256, N)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.W1_txt.weight)
        nn.init.constant_(self.W1_txt.bias, 0)
        nn.init.xavier_normal_(self.W1_sen.weight)
        nn.init.constant_(self.W1_sen.bias, 0)
        nn.init.xavier_normal_(self.W1_vis.weight)
        nn.init.constant_(self.W1_vis.bias, 0)
        nn.init.xavier_normal_(self.W2_vis.weight)
        nn.init.constant_(self.W2_vis.bias, 0)
        nn.init.xavier_normal_(self.W_fc_pos.weight)
        nn.init.constant_(self.W_fc_pos.bias, 0)
        nn.init.xavier_normal_(self.W_fc.weight)
        nn.init.constant_(self.W_fc.bias, 0)


    # def sinkhorn(self, x):
        # x = torch.exp(x / self.tau)  # exp initialize

        # # simple sinkhorn
        # for _ in range(self.n_iters):
        #     x = x / (10e-8 + torch.sum(x, -2, keepdim=True))
        #     x = x / (10e-8 + torch.sum(x, -1, keepdim=True))
        # return x

    def sinkhorn(self, x):
        x = torch.exp(x / self.tau)  # exp initialize

        for i in range(self.n_iters):
            x = x - (torch.logsumexp(x, dim=-1, keepdim=True))
            x = x - (torch.logsumexp(x, dim=-2, keepdim=True))
        return torch.exp(x)

    def forward(self, seq):
        x_txt = seq[:, :, :512]  # 512
        x_vis = seq[:, :, 512:2560]  # 2048
        x_pos = seq[:, :, 2560:2564]  # 4
        x_sen = seq[:, :, 2564:]  # 512
        x_txt = F.relu(self.W1_txt(x_txt))
        x_vis = F.relu(self.W1_vis(x_vis))
        x_vis = F.relu(self.W2_vis(x_vis))
        x_sen = F.relu(self.W1_sen(x_sen))
        x_pos = F.relu(x_pos)
        x = torch.cat((x_txt, x_vis, x_pos, x_sen), dim=-1)  # [batch_size, N, fea]
        x = F.relu(self.W_fc_pos(x))
        x = torch.tanh(self.W_fc(x))

        return self.sinkhorn(x)  # [batch, N, N]


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, vocab_size, embed_dim=512, decoder_dim=512, vision_dim=768, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.fix_len = 3

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # ``````````````` layer
        self.visualBERT = VisualBertModel.from_pretrained("uclanlp")
        
        self.attention_obj = ObjectAttention(embed_dim, decoder_dim, decoder_dim)

        # self.answer_lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True, dropout=dropout,
        #                            bidirectional=False)

        self.sentence_lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True, dropout=dropout, bidirectional=False)
        self.onehot_lstm = TextEncoder(input_size=embed_dim, hidden_size=embed_dim, dropout=dropout)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)
        self.embedding_dropout = nn.Dropout(self.dropout)

        self.one_hot_decoder = Decoder(decoder_dim, embed_dim, vision_dim, vocab_size, dropout)
        # self.one_hot_decoder_1 = Decoder(decoder_dim, embed_dim, vision_dim, vocab_size, dropout)
        self.multi_decoder = MultiDecoder(decoder_dim, embed_dim, vision_dim, vocab_size, dropout)

        self.sorting_network = SinkhornNetwork(self.fix_len, 20, 0.6)  # sorting network for objects

        # self.tokenizer =

        self.init_weights()  # initialize some layers with the uniform distribution

        self.W1_ig = nn.Linear(decoder_dim, decoder_dim)
        self.W1_hg = nn.Linear(decoder_dim, decoder_dim)
        self.att_g = nn.Linear(decoder_dim, 1, bias=False)

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def calculate_id(self, h1, ctrl_det_idxs, det_curr_emb):
        # det_curr = torch.gather(det_curr_emb, 1, ctrl_det_idxs.unsqueeze(1))
        g_gate = torch.sigmoid(self.W1_ig(h1) + self.W1_hg(det_curr_emb.squeeze(1)))
        g_gate = self.att_g(g_gate).squeeze(-1)
        g_t = (g_gate > 0.15).int()
        ctrl_det_idxs = g_t + ctrl_det_idxs
        ctrl_det_idxs = torch.clamp(ctrl_det_idxs, 0, 2)
        return ctrl_det_idxs

    def obj_tranform(self, soft_perm_matrix, encoded_objects):
        predict_matrix = F.one_hot(soft_perm_matrix.argmax(dim=-1), encoded_objects.shape[1]).float()
        return torch.matmul(predict_matrix, encoded_objects.unsqueeze(2).float()).long()


    def forward(self, visual_embeds, visual_token_type_ids, visual_attention_mask,
                sentence_gt, sentence_gt_length, sentence_inputs_input_ids, sentence_inputs_token_type_ids,
                sentence_inputs_mask, sentence_input, obj, obj_vis, obj_pos, gt_obj=None, teacher_forcing=False, teacher_forcing_ratio=1.0):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_sentences: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param sentence_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        # self.answer_lstm.flatten_parameters()

        # answers_embedding = self.embedding_dropout(self.embedding(answer))
        # answers_lstm, (answer_h1, answer_c1) = self.answer_lstm(answers_embedding)
        # h1, c1 = answer_h1.squeeze(0), answer_c1.squeeze(0)

        self.sentence_lstm.flatten_parameters()
        sentences_input_embedding = self.embedding_dropout(self.embedding(sentence_input))
        sentences_lstm, (sentence_input_h1, sentence_input_c1) = self.sentence_lstm(sentences_input_embedding)
        h1, c1 = sentence_input_h1.squeeze(0), sentence_input_c1.squeeze(0)
        
        ctrl_det_idxs = torch.zeros((visual_embeds.shape[0], ), requires_grad=True).long().cuda()  # [b, ] detection ids

        
        objects_embedding = self.embedding(obj)

        seq = torch.cat((objects_embedding, obj_vis, obj_pos, sentences_lstm[:, -objects_embedding.shape[1]:, :]), dim=-1)

        soft_perm_matrix = self.sorting_network(seq)

        soft_perm_matrix = torch.transpose(soft_perm_matrix, 1, 2)

        if teacher_forcing:
            obj_new_list = gt_obj
        else:
            obj_new_list = self.obj_tranform(soft_perm_matrix, obj).squeeze(-1)

        new_objects_embedding = self.embedding(obj_new_list)



        fusion_feature = self.visualBERT(input_ids=sentence_inputs_input_ids,
                                         token_type_ids=sentence_inputs_token_type_ids,
                                         attention_mask=sentence_inputs_mask,
                                         visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask,
                                         visual_token_type_ids=visual_token_type_ids).last_hidden_state
        # print(f'fusion_feature.shape:', fusion_feature.shape)
        sentence_embed = self.embedding_dropout(self.embedding(sentence_gt))   # (batch_size, max_caption_length, embed_dim)

        bs, max_q_len, _ = sentence_embed.size()
        # simple sentence
        outputs = torch.zeros(bs, max_q_len, self.vocab_size).cuda()
        simple_sentence = torch.zeros(bs, max_q_len).cuda()
        output = sentence_embed[:, 0]
        for i in range(1):
            # generate simple sentence
            for t in range(1, max_q_len):
                if t > 1:
                    ctrl_det_idxs = self.calculate_id(h1, ctrl_det_idxs, det_curr_emb)
                det_curr = torch.gather(obj_new_list, 1, ctrl_det_idxs.unsqueeze(1))
                # if the object is padding, move to another object
                zero_ind = (det_curr == 0).int().squeeze(-1)
                ctrl_det_idxs = ctrl_det_idxs + zero_ind
                ctrl_det_idxs = torch.clamp(ctrl_det_idxs, 0, 2)
                det_curr = torch.gather(obj_new_list, 1, ctrl_det_idxs.unsqueeze(1))

                det_curr_emb = self.embedding(det_curr)  # [b, 1, 512]
                output, h1, c1 = self.one_hot_decoder(output, h1, c1, fusion_feature, det_curr_emb)
                outputs[:, t] = output
                simple_sentence[:, t] = torch.argmax(output, dim=1)
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.max(1)[1]
                top1 = self.embedding_dropout(self.embedding(top1))
                output = sentence_embed[:, t] if teacher_force else top1

            sentences_input_embedding = self.embedding_dropout(self.embedding(simple_sentence.long()))
            sentences_lstm, (sentence_input_h1, sentence_input_c1) = self.sentence_lstm(sentences_input_embedding)
            h1, c1 = sentence_input_h1.squeeze(0), sentence_input_c1.squeeze(0)

            # Create tensors to hold word predicion scores
            outputs = torch.zeros(bs, max_q_len, self.vocab_size).cuda()
            output = sentence_embed[:, 0]

        # generate one hot sentence
        for t in range(1, max_q_len):
            if t > 1:
                ctrl_det_idxs = self.calculate_id(h1, ctrl_det_idxs, det_curr_emb)
            det_curr = torch.gather(obj_new_list, 1, ctrl_det_idxs.unsqueeze(1))
            # if the object is padding, move to another object
            zero_ind = (det_curr == 0).int().squeeze(-1)
            ctrl_det_idxs = ctrl_det_idxs + zero_ind
            ctrl_det_idxs = torch.clamp(ctrl_det_idxs, 0, 2)
            det_curr = torch.gather(obj_new_list, 1, ctrl_det_idxs.unsqueeze(1))

            det_curr_emb = self.embedding(det_curr)  # [b, 1, 512]
            output, h1, c1 = self.one_hot_decoder(output, h1, c1, fusion_feature, det_curr_emb)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            top1 = self.embedding_dropout(self.embedding(top1))
            output = sentence_embed[:, t] if teacher_force else top1

        enc_sim_phrase = self.lstm(sentence_embed)[0][:, -1, :]
        enc_out = self.lstm(self.embedding(outputs.max(2)[0].long()))[0][:, -1, :]

        return outputs, soft_perm_matrix, enc_out, enc_sim_phrase
