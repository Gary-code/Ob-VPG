import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5, 0'

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from models import DecoderWithAttention
from datasets import get_loader
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import config
import random
import numpy as np
from tqdm import tqdm
from evaluation import metric
from optimization import BertAdam


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Data parameters
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 150  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, data_name, word_map

    seed_everything(config.seed)
    # Read word map
    with open(config.vocabulary_path, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    decoder = DecoderWithAttention(
        embed_dim=emb_dim,
        decoder_dim=decoder_dim,
        vocab_size=len(word_map),
        dropout=dropout)
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            print(name)

    # torch.distributed.init_process_group(backend="nccl")
    decoder = nn.DataParallel(decoder, device_ids=[0, 1]).cuda()
    # decoder = decoder.cuda()
    # decoder = nn.parallel.DistributedDataParallel(decoder)
    # decoder = decoder.cuda()

    if checkpoint is not None:
        print('load checkpoint')
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder.load_state_dict(checkpoint['decoder'])

    # Move to GPU, if available
    # decoder = decoder.cuda()

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss(reduction='sum').to(device)  # sentence loss
    # criterion_dis = nn.MultiLabelMarginLoss().to(device)
    criterion_sort = nn.MSELoss(reduction='sum').to(device)  # sort loss
    criterion_dis = JointEmbeddingLoss
    # Custom dataloaders
    train_loader = get_loader('train', config.train_spatial_feature_path)
    valid_loader = get_loader('valid', config.dev_spatial_feature_path)
    # test_loader = get_loader('test', config.test_spatial_feature_path)
    batch_per_epoch = len(train_loader)
    t_total = int(batch_per_epoch * epochs)
    named_parameters = list(decoder.named_parameters())
    parameters = [
        {'params': [p for n, p in named_parameters if 'visualBERT' in n], 'lr': 1e-4},
        {'params': [p for n, p in named_parameters if 'visualBERT' not in n], 'lr': 2e-3},
    ]
    decoder_optimizer = BertAdam(parameters,
                                 lr=2e-3,
                                 warmup=0.1,
                                 t_total=t_total)
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 100:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              decoder=decoder,
              criterion_ce=criterion_ce,
              criterion_sort=criterion_sort,
              criterion_dis=criterion_dis,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4, references, hypotheses = validate(val_loader=valid_loader,
                                                        decoder=decoder,
                                                        criterion_ce=criterion_ce,
                                                        criterion_dis=criterion_dis,
                                                        criterion_sort=criterion_sort,
                                                        word_map=word_map)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0
            print('save best checkpoint')
            save_checkpoint(data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer, recent_bleu4,
                            is_best)
            idx2word = {index: word for word, index in word_map.items()}

            print('writing best file')
            with open('true.txt', 'w') as f:
                for r in tqdm(references):
                    words = [idx2word[i] for i in r]
                    while '<end>' in words:
                        words.remove('<end>')
                    while '<start>' in words:
                        words.remove('<start>')
                    while '<pad>' in words:
                        words.remove('<pad>')
                    f.write(' '.join(words) + '\n')

            with open('pred.txt', 'w') as f:
                for r in tqdm(hypotheses):
                    words = [idx2word[i] for i in r]
                    while '<end>' in words:
                        words.remove('<end>')
                    while '<start>' in words:
                        words.remove('<start>')
                    while '<pad>' in words:
                        words.remove('<pad>')
                    f.write(' '.join(words) + '\n')


# validate(val_loader=valid_loader,
#          decoder=decoder,
#          criterion_ce=criterion_ce,
#          criterion_dis=criterion_dis,
#          word_map=word_map)
# validate(val_loader=test_loader,
#          decoder=decoder,
#          criterion_ce=criterion_ce,
#          criterion_dis=criterion_dis,
#          word_map=word_map)

def train(train_loader, decoder, criterion_ce, criterion_sort, criterion_dis, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    count = 0
    # Batches
    for i, (visual_embeds, visual_token_type_ids, visual_attention_mask,
            sentence_gt, sentence_gt_length, sentence_inputs_input_ids,
            sentence_inputs_token_type_ids, sentence_inputs_mask, sentence_input, obj, gt_obj, obj_vis, obj_pos,
            hard_perm) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        var_params = {
            'requires_grad': False,
        }
        obj = obj.cuda()
        gt_obj = gt_obj.cuda()
        obj_vis = obj_vis.cuda()
        obj_pos = obj_pos.cuda()
        sentence_input = sentence_input.cuda()
        visual_embeds = Variable(visual_embeds.cuda(), **var_params)
        hard_perm = Variable(hard_perm.cuda(), **var_params)
        visual_token_type_ids = Variable(visual_token_type_ids.cuda(), **var_params)
        visual_attention_mask = Variable(visual_attention_mask.cuda(), **var_params)
        sentence_gt = Variable(sentence_gt.cuda(), **var_params)
        sentence_gt_length = Variable(sentence_gt_length.cuda(), **var_params)
        sentence_inputs_input_ids = Variable(sentence_inputs_input_ids.cuda(), **var_params)
        sentence_inputs_token_type_ids = Variable(sentence_inputs_token_type_ids.cuda(), **var_params)
        sentence_inputs_mask = Variable(sentence_inputs_mask.cuda(), **var_params)

        # Forward prop.
        outputs, soft_perm, enc_out, enc_sim_phrase = decoder(visual_embeds, visual_token_type_ids,
                                                              visual_attention_mask,
                                                              sentence_gt, sentence_gt_length,
                                                              sentence_inputs_input_ids, sentence_inputs_token_type_ids,
                                                              sentence_inputs_mask, sentence_input, obj, obj_vis,
                                                              obj_pos, gt_obj, teacher_force=True)

        # Max-pooling across predicted words across time steps for discriminative supervision
        sentence_gt_length = (sentence_gt_length - 1).squeeze(dim=-1).cpu().numpy().tolist()
        targets = sentence_gt[:, 1:]

        outputs = pack_padded_sequence(outputs[:, 1:, :], sentence_gt_length, batch_first=True,
                                       enforce_sorted=False).data
        targets = pack_padded_sequence(targets, sentence_gt_length, batch_first=True, enforce_sorted=False).data
        loss1 = criterion_ce(outputs, targets)
        loss2 = criterion_sort(soft_perm, hard_perm)
        loss3 = criterion_dis(enc_out, enc_sim_phrase)

        # print(f'loss1, loss2', loss1, loss2)

        loss = loss1 + loss2 * 10 + loss3 * 10
        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(outputs, targets, 5)
        losses.update(loss.item(), sum(sentence_gt_length))
        top5accs.update(top5, sum(sentence_gt_length))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
            print(f'loss1 & loss2 & loss3:', loss1, loss2, loss3)
            count += 1


def validate(val_loader, decoder, criterion_ce, criterion_sort, criterion_dis, word_map):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    with torch.no_grad():
        for i, (visual_embeds, visual_token_type_ids, visual_attention_mask,
                sentence_gt, sentence_gt_length, sentence_inputs_input_ids,
                sentence_inputs_token_type_ids, sentence_inputs_mask, sentence_input, obj, gt_obj, obj_vis, obj_pos,
                hard_perm) in enumerate(val_loader):

            obj_vis, obj_pos = obj_vis.cuda(), obj_pos.cuda()
            obj = obj.cuda()
            gt_obj = gt_obj.cuda()
            sentence_input = sentence_input.cuda()
            # Move to GPU, if available
            var_params = {
                'requires_grad': False,
            }
            visual_embeds = Variable(visual_embeds.cuda(), **var_params)
            hard_perm = Variable(hard_perm.cuda(), **var_params)
            visual_token_type_ids = Variable(visual_token_type_ids.cuda(), **var_params)
            visual_attention_mask = Variable(visual_attention_mask.cuda(), **var_params)
            sentence_gt = Variable(sentence_gt.cuda(), **var_params)
            sentence_gt_length = Variable(sentence_gt_length.cuda(), **var_params)
            sentence_inputs_input_ids = Variable(sentence_inputs_input_ids.cuda(), **var_params)
            sentence_inputs_token_type_ids = Variable(sentence_inputs_token_type_ids.cuda(), **var_params)
            sentence_inputs_mask = Variable(sentence_inputs_mask.cuda(), **var_params)

            # Forward prop.
            outputs, soft_perm, enc_out, enc_sim_phrase = decoder(visual_embeds, visual_token_type_ids,
                                                                  visual_attention_mask,
                                                                  sentence_gt, sentence_gt_length,
                                                                  sentence_inputs_input_ids,
                                                                  sentence_inputs_token_type_ids,
                                                                  sentence_inputs_mask, sentence_input, obj, obj_vis,
                                                                  obj_pos)

            # Max-pooling across predicted words across time steps for discriminative supervision
            sentence_gt_length = (sentence_gt_length - 1).squeeze().cpu().numpy().tolist()
            targets = sentence_gt[:, 1:]
            outputs_copy = outputs.clone()
            outputs = pack_padded_sequence(outputs[:, 1:, :], sentence_gt_length, batch_first=True,
                                           enforce_sorted=False).data
            targets = pack_padded_sequence(targets, sentence_gt_length, batch_first=True, enforce_sorted=False).data
            loss1 = criterion_ce(outputs, targets)
            loss2 = criterion_sort(soft_perm, hard_perm)
            loss3 = criterion_dis(enc_out, enc_sim_phrase)

            loss = loss1 + loss2 * 10 + loss3 * 10

            # Keep track of metrics
            losses.update(loss.item(), sum(sentence_gt_length))
            top5 = accuracy(outputs, targets, 5)
            top5accs.update(top5, sum(sentence_gt_length))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))
                print(f'loss1 & loss2 & loss3:', loss1, loss2, loss3)

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            for j in range(sentence_gt.shape[0]):
                img_ques = sentence_gt[j].cpu().numpy().tolist()
                img_sentence_gts = [w for w in img_ques if
                                    w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                references.append(img_sentence_gts)

            # Hypotheses
            _, preds = torch.max(outputs_copy, dim=-1)  # [batch_size, len]
            preds = preds.tolist()
            hypotheses.extend(preds)
            assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores

    references_2 = []
    hypotheses_2 = []
    idx2word = {index: word for word, index in word_map.items()}

    # calculate bleu

    for r in tqdm(references):
        words = [idx2word[i] for i in r]
        while '<end>' in words:
            words.remove('<end>')
        while '<start>' in words:
            words.remove('<start>')
        while '<pad>' in words:
            words.remove('<pad>')
        references_2.append([' '.join(words)])

    for r in tqdm(hypotheses):
        words = [idx2word[i] for i in r]
        while '<end>' in words:
            words.remove('<end>')
        while '<start>' in words:
            words.remove('<start>')
        while '<pad>' in words:
            words.remove('<pad>')
        hypotheses_2.append([' '.join(words)])

    gen = {i: s for i, s in enumerate(hypotheses_2)}
    ref = {i: s for i, s in enumerate(references_2)}
    print(hypotheses_2[:5])
    print(references_2[:5])

    bleu = metric(ref, gen)
    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}\n'.format(
            loss=losses,
            top5=top5accs))
    # print('bleu: %s, cider: %.6s, meteor: %.6s, rouge: %.6s' % (bleu, None, None, None))
    print('bleu: %s' % bleu)
    return bleu[-1][-1], references, hypotheses


if __name__ == '__main__':
    main()
