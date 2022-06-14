import json
import os
import os.path
import re

import _pickle as cPickle
from PIL import Image
import h5py
import torch
import torch.utils.data as data
import numpy as np
from collections import Counter
import config
from spacy.tokenizer import Tokenizer
import en_core_web_sm
from tqdm import tqdm
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def get_loader(mode, features_file):
    """ Returns a data loader for the desired split """
    if mode == 'train':
        data_path = config.train_data_path
    elif mode == 'valid':
        data_path = config.dev_data_path
    elif mode == 'test':
        data_path = config.test_data_path
        config.batch_size = 1
    split = COCOSentencesSet(data_path, features_file)
    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config.batch_size,
        shuffle=True if mode == 'train' else False,  # only shuffle the data in training
        pin_memory=True,
        num_workers=8,
        # collate_fn=collate_fn,
    )
    return loader


def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)


class COCOSentencesSet(data.Dataset):
    """ VQA dataset, open-ended """

    def __init__(self, data_path, features_file, fix_len=3):
        super(COCOSentencesSet, self).__init__()
        self.fix_len = fix_len
        with open(data_path, 'r') as fd:
            data_json = json.load(fd)
        with open(config.vocabulary_path, 'r') as fd:
            vocab_json = json.load(fd)

        self.classes = []
        with open(config.classes_path, 'r') as f:
            for obj in f.readlines():
                self.classes.append(obj.split(',')[0].lower().strip())


        self.sentence_ids = [i["input_id"] for i in data_json]

        # obj
        self.objects = [i["sentence1_entities"] for i in data_json]
        self.gt_objects = [i["sentence2_entities"] for i in data_json]

        # vocab
        self.vocab = vocab_json
        self.token_to_index = self.vocab
        # self.answer_to_index = self.vocab

        # q and a
        self.sentence_gts, self.sentence_inputs, self.sentence_oris, self.gt_objects_list = prepare(data_json) ############################################  q_type
        print('max_sentence_length : {}'.format(self.max_sentence_length))

        self.sentence_gts = [self._encode_sentence(q, self.max_sentence_length, if_q=True) for q in self.sentence_gts]
        # self.answers = [self._encode_answers(a) for a in self.answers]
        self.sentence_oris = [self._encode_sentence(q, self.max_sentence_length, if_q=True) for q in self.sentence_oris]
        # self.sentence_inputs = [self._encode_sentence(q, self.max_sentence_length, if_q=True) for q in self.sentence_inputs]
        self.sentence_inputs = tokenizer(self.sentence_inputs, return_tensors="pt", padding=True)
        #############################################################################question_type
        # v
        self.image_features_path = features_file
        self.coco_id_to_index = self._create_coco_id_to_index()
        self.coco_ids = [d['image_id'] for d in data_json]

    @property
    def max_sentence_length(self):
        if not hasattr(self, '_max_length'):
            data_max_length1 = max(map(len, self.sentence_oris))
            print(f'data_max_length1', data_max_length1)
            data_max_length2 = max(map(len, self.sentence_gts))
            print(f'data_max_length2', data_max_length2)
            data_max_length = max(data_max_length1, data_max_length2)
            # data_max_length = data_max_length2
            self._max_length = min(config.max_q_length, data_max_length)
        return self._max_length

    @property
    def num_tokens(self):
        return len(self.token_to_index)

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        if not hasattr(self, 'features_file'):
            self.features_file = h5py.File(self.image_features_path, 'r')
        coco_ids = self.features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def _encode_sentence(self, sentence, max_length, if_q=False):
        """ Turn a question into a vector of indices and a question length """
        if if_q:
            enc_q = [self.token_to_index['<start>']] + [self.token_to_index.get(word, self.token_to_index['<unk>']) for
                                                        word in sentence] + [
                        self.token_to_index['<end>']] + [self.token_to_index['<pad>']] * (max_length - len(sentence))
            # Find questions lengths
            q_len = len(sentence) + 2
            vec = torch.LongTensor(enc_q)
        else:
            enc_q = [self.token_to_index.get(word, self.token_to_index['<unk>']) for word in sentence] + [
                self.token_to_index['<pad>']] * (max_length - len(sentence))
            # Find questions lengths
            q_len = len(sentence)
            vec = torch.LongTensor(enc_q)
        return vec, torch.LongTensor([q_len])

    def _encode_objects(self, objects):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        enc_a = [self.token_to_index.get(word, self.token_to_index['<unk>']) for word in objects] + [self.token_to_index['<pad>']] * (self.fix_len - len(objects))
        # Find questions lengths
        a_len = len(objects)
        vec = torch.LongTensor(enc_a[:self.fix_len])
        return vec, torch.LongTensor([a_len])

    def _encode_answers(self, answers):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        enc_a = [self.answer_to_index.get(word, self.answer_to_index['<unk>']) for word in answers] + [
            self.answer_to_index['<pad>']] * (config.max_a_length - len(answers))
        # Find questions lengths
        a_len = len(answers)
        vec = torch.LongTensor(enc_a[:config.max_a_length])
        return vec, torch.LongTensor([a_len])

    def _load_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')
        index = self.coco_id_to_index[int(image_id)]
        img = self.features_file['features'][index]
        bboxes = self.features_file['boxes'][index]
        widths = self.features_file['widths'][index]
        heights = self.features_file['heights'][index]
        clses = self.features_file['objects_id'][index]
        # clses = clses.split(';')
        return img, bboxes, widths, heights, clses

    def __getitem__(self, item):
        s_gt, s_gt_length = self.sentence_gts[item]
        s_ori, s_ori_length = self.sentence_oris[item]
        o = self.objects[item]
        o_gt = self.gt_objects[item]

        o = o[:self.fix_len]
        o_gt = o_gt[:self.fix_len]

        object_encode, _ = self._encode_objects(o)
        gt_obj_encode, _ = self._encode_objects(o_gt)


        # a, a_length = self.answers[item]
        image_id = self.coco_ids[item]
        visual_embeds, bboxes, widths, heights, clses = self._load_image(image_id)
        visual_embeds = torch.from_numpy(visual_embeds).float()
        bboxes = bboxes.T
        visual_embeds = visual_embeds.T
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)

        det_sequences_visual = torch.zeros((self.fix_len, visual_embeds.shape[-1]))
        det_sequences_position = torch.zeros((self.fix_len, 4))
        hard_perm_matrix = torch.zeros((self.fix_len, self.fix_len))

        det_ind = torch.full((self.fix_len, 1), -1, dtype=torch.float)
        gt_ind = torch.full((self.fix_len, 1), -1, dtype=torch.float)

        for j, (det_class, gt_class) in enumerate(zip(o, o_gt)):
            if gt_class in o:
                hard_perm_matrix[j][o.index(gt_class)] = 1

            # if det_class in self.vectors:
                # det_sequences_word[j] = torch.from_numpy(self.vectors[det_class])

            if det_class not in self.classes:
                continue
            object_id = self.classes.index(det_class)
            det_ind[j][0] = object_id
            det_ids = [i for i, j in enumerate(clses) if j == object_id]
            if len(det_ids) == 0:
                continue
            det_sequences_visual[j] = visual_embeds[det_ids[0]]
            bbox = bboxes[det_ids[0]]
            det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / widths
            det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / heights
            det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / widths
            det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / heights

            if gt_class not in self.classes:
                continue
            object_id = self.classes.index(gt_class)
            gt_ind[j][0] = object_id


        sentence_inputs_input_ids, sentence_inputs_token_type_ids, sentence_inputs_mask = \
            self.sentence_inputs['input_ids'][item], self.sentence_inputs['token_type_ids'][item], \
            self.sentence_inputs['attention_mask'][item]

        return visual_embeds, visual_token_type_ids, visual_attention_mask, s_gt, s_gt_length, sentence_inputs_input_ids, \
               sentence_inputs_token_type_ids, sentence_inputs_mask, s_ori, object_encode, gt_obj_encode, det_sequences_visual, det_sequences_position, hard_perm_matrix

    def __len__(self):
        return len(self.sentence_gts)


# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def prepare(data):
    print('tokenizer questions and answers ...')
    sentences_gt, sentences_input, sentences_ori, gt_objects_list = [], [], [], []
    nlp = en_core_web_sm.load()
    tokenizer = Tokenizer(nlp.vocab)
    for i, row in tqdm(enumerate(data)):
        sentence_gt = row['sentence2'].strip().lower()
        # sentence_gt = _special_chars.sub('', sentence_gt)
        sentence_input = row['sentence1'].strip().lower() + ' '.join(row['sentence1_entities'])
        sentence_ori = [t.text if '.' not in t.text else t.text[:-1] for t in tokenizer(sentence_input)]
        sentence_gt = [t.text if '.' not in t.text else t.text[:-1] for t in tokenizer(sentence_gt)]
        gt_object = ' '.join(row["sentence2_entities"])
        if i < 3:
            print(sentence_gt)
            print(sentence_input)
        sentences_gt.append(sentence_gt)
        sentences_input.append(sentence_input)
        sentences_ori.append(sentence_ori)
        gt_objects_list.append(gt_object)
    # print(f'len:', len(sentences_input), sentences_input[1], len(sentences_input[1]))
    return sentences_gt, sentences_input, sentences_ori, gt_objects_list


def prepare_answers(answers_json):
    """ Normalize answers from a given answer json in the usual VQA format. """
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]

    # The only normalization that is applied to both machine generated answers as well as
    # ground truth answers is replacing most punctuation with space (see [0] and [1]).
    # Since potential machine generated answers are just taken from most common answers, applying the other
    # normalizations is not needed, assuming that the human answers are already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96

    def process_punctuation(s):
        # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
        # this version should be faster since we use re instead of repeated operations on str's
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()

    for answer_list in answers:
        answer = list(map(process_punctuation, answer_list))
        counter = Counter(answer)
        word, freq = counter.most_common(1)[0]
        if freq > 1:
            yield word.split()
        else:
            yield answer[0].split()


class CocoImages(data.Dataset):
    """ Dataset for MSCOCO images located in a folder on the filesystem """

    def __init__(self, path, transform=None):
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)
