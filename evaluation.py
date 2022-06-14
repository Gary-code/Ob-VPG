from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.spice.spice import Spice
import json

# with open('examples/gts.json', 'r') as file:
#     gts = json.load(file)
# with open('examples/res.json', 'r') as file:
#     res = json.load(file)

def bleu(gts, res):
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)

    print('belu = %s' % score)
    return score

def cider(gts, res):
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    print('cider = %s' % score)
    return score

def meteor(gts, res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print('meter = %s' % score)
    return score

def rouge(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print('rouge = %s' % score)
    return score

def spice(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print('spice = %s' % score)

def metric(gts, res):

    bleu_score = bleu(gts, res)
    # cider_score = cider(gts, res)
    # meteor_score = meteor(gts, res)
    # rouge_score = rouge(gts, res)
    # spice()

    return bleu_score,
           # cider_score, meteor_score, rouge_score


def idx2question(idx, idx_ref, question_vocab):
    gen_sentences = []
    ref_sentences = []

    for i in range(idx.shape[0]):
        words = [str(question_vocab.get(int(index))) for index in idx[i].tolist()]
        while 'EOS' in words:
            words.remove('EOS')
        while 'PAD' in words:
            words.remove('PAD')
        sentence = [' '.join(words) + '.']
        gen_sentences.append(sentence)

    gen = {i:s for i,s in enumerate(gen_sentences)}

    for i in range(idx_ref.shape[0]):
        words = [str(question_vocab.get(int(index))) for index in idx_ref[i].tolist()]
        while 'EOS' in words:
            words.remove('EOS')
        while 'START' in words:
            words.remove('START')
        while 'PAD' in words:
            words.remove('PAD')
        sentence = [' '.join(words) + '.']
        ref_sentences.append(sentence)

    ref = {i:s for i,s in enumerate(ref_sentences)}

    return gen, ref
