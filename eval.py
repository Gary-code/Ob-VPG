import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
from torch.autograd import Variable
from models import DecoderWithAttention
import torch.nn.functional as F
from tqdm import tqdm
from datasets import get_loader
import random
import numpy as np
import config
import json
from evaluation import metric


# Parameters
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint_file = 'best_checkpoint.pth.tar'  # model checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
emb_dim = 512  # dimension of word embeddings
attention_dim = 1024  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.3


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

with open(config.vocabulary_path, 'r') as j:
    word_map = json.load(j)

seed_everything(config.seed)


# Load model
torch.nn.Module.dump_patches = True
checkpoint = torch.load(checkpoint_file, map_location = device)
decoder = DecoderWithAttention(
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
# decoder = nn.DataParallel(decoder, device_ids=[0, 1]).cuda()
decoder = decoder.cuda()
decoder.load_state_dict(checkpoint['decoder'])
decoder.eval()

# decoder = decoder.module
# Load word map (word2ix)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """
    # DataLoader
    loader = get_loader('test', config.test_spatial_feature_path)
    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    count = 0
    # For each image
    with torch.no_grad():
        for i, (visual_embeds, visual_token_type_ids, visual_attention_mask,
               tripple_input_ids, tripple_token_type_ids, tripple_mask,
               onehot_tripple_input_ids, onehot_tripple_token_type_ids, onehot_tripple_mask,
               answer, one_hot_q, one_hot_q_length, ques, q_length) in enumerate(tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

            k = beam_size

            # Move to GPU device, if available
            var_params = {
                'requires_grad': False,
            }
            visual_embeds = Variable(visual_embeds.cuda(), **var_params)
            visual_token_type_ids = Variable(visual_token_type_ids.cuda(), **var_params)
            visual_attention_mask = Variable(visual_attention_mask.cuda(), **var_params)
            tripple_input_ids = Variable(tripple_input_ids.cuda(), **var_params)
            tripple_mask = Variable(tripple_mask.cuda(), **var_params)
            tripple_token_type_ids = Variable(tripple_token_type_ids.cuda(), **var_params)
            onehot_tripple_input_ids = Variable(onehot_tripple_input_ids.cuda(), **var_params)
            onehot_tripple_token_type_ids = Variable(onehot_tripple_token_type_ids.cuda(), **var_params)
            onehot_tripple_mask = Variable(onehot_tripple_mask.cuda(), **var_params)
            answer = Variable(answer.cuda(), **var_params)
            one_hot_q = Variable(one_hot_q.cuda(), **var_params)
            one_hot_q_length = Variable(one_hot_q_length.cuda(), **var_params)
            ques = Variable(ques.cuda(), **var_params)
            q_length = Variable(q_length.cuda(), **var_params)


            answers_embedding = decoder.embedding(answer)
            answers_lstm, (answer_h1, answer_c1) = decoder.answer_lstm(answers_embedding)
            h1, c1 = answer_h1.squeeze(0), answer_c1.squeeze(0) # [1, 512]

            fusion_feature = decoder.visualBERT(input_ids=onehot_tripple_input_ids,
                                             token_type_ids=onehot_tripple_token_type_ids,
                                             attention_mask=onehot_tripple_mask,
                                             visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask,
                                             visual_token_type_ids=visual_token_type_ids).last_hidden_state


            outputs = []
            k_prev_words = torch.LongTensor([[word_map['<start>']]]).cuda() # (1, 1)
            output = decoder.embedding(k_prev_words).squeeze(1)  # (1, embed_dim)
            for t in range(28):
                output, h1, c1 = decoder.one_hot_decoder(output, h1, c1, fusion_feature)
                top1 = output.max(1)[1]
                if top1.cpu().item() == word_map['<end>']:
                    # print('break')
                    break
                outputs.append(top1)
                top1 = decoder.embedding(top1)
                output = top1

            one_hot_q_len = torch.LongTensor([[len(outputs)]]).cuda()
            outputs = torch.stack(outputs, 1)

            one_hot_q_embed = decoder.embedding(outputs)

            one_hot_q_feature = decoder.onehot_lstm(one_hot_q_embed, one_hot_q_len)
            h1, c1 = answer_h1.squeeze(0), answer_c1.squeeze(0) # [1, 512]

            fusion_feature = decoder.visualBERT(input_ids=tripple_input_ids,
                                             token_type_ids=tripple_token_type_ids,
                                             attention_mask=tripple_mask,
                                             visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask,
                                             visual_token_type_ids=visual_token_type_ids).last_hidden_state

            one_hot_q_feature = one_hot_q_feature.expand(k, one_hot_q_feature.size(1), one_hot_q_feature.size(2))
            h1 = h1.expand(k, 512)
            c1 = c1.expand(k, 512)
            one_hot_q_len = one_hot_q_len.expand(k, one_hot_q_len.size(1))
            fusion_feature = fusion_feature.expand(k, fusion_feature.size(1), fusion_feature.size(2))


            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).cuda() # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).cuda()  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            # h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)

            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                scores, h1, c1 = decoder.multi_decoder(embeddings, h1, c1, fusion_feature, one_hot_q_feature, one_hot_q_len)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                h1 = h1[prev_word_inds[incomplete_inds]]
                c1 = c1[prev_word_inds[incomplete_inds]]
                fusion_feature = fusion_feature[prev_word_inds[incomplete_inds]]
                one_hot_q_len = one_hot_q_len[prev_word_inds[incomplete_inds]]
                one_hot_q_feature = one_hot_q_feature[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 28:
                    complete_seqs_scores.extend(top_k_scores.cpu().squeeze(1).numpy().tolist())
                    complete_seqs.extend(seqs.cpu().numpy().tolist())
                    count += 1
                    break
                step += 1
            # print(complete_seqs_scores)
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

            # References
            for j in range(ques.shape[0]):
                img_ques = ques[j].tolist()
                img_questions = [rev_word_map[w] for w in img_ques if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                references.append([' '.join(img_questions)])

            # Hypotheses
            hypothesis = [rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            #print(hypothesis)
            hypotheses.append([' '.join(hypothesis)])
            assert len(references) == len(hypotheses)


    print(' > 28 size : {}'.format(count))
    # Calculate scores
    gen = {i: s for i, s in enumerate(hypotheses)}
    ref = {i: s for i, s in enumerate(references)}

    bleu, cider, meteor, rouge = metric(ref, gen)
    print('bleu: %s, cider: %.6s, meteor: %.6s, rouge: %.6s' % (bleu, cider, meteor, rouge))
    output = []
    for y_pred, y_true in zip(hypotheses, references):
        output.append({
            'pred': y_pred[0],
            'true': y_true[0]
        })
    with open('predict2.json', 'w') as fd:
        json.dump(output, fd, indent=4)



if __name__ == '__main__':
    beam_size = 3
    evaluate(beam_size)
