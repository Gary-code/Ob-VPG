# paths
mode = 'train'
path = './data/'
qa_path = path + 'split/'  # directory containing the question and annotation jsons
train_spatial_feature_path = '/home/cike/x_VQG/genome-trainval.h5'
dev_spatial_feature_path = '/home/cike/x_VQG/genome-trainval.h5'
test_spatial_feature_path = '/home/cike/x_VQG/genome-trainval.h5'
# train_spatial_feature_path = path + 'faster_rcnn_train.h5'
# dev_spatial_feature_path = path + 'faster_rcnn_dev.h5'
# test_spatial_feature_path = path + 'faster_rcnn_test.h5'    ########.h5文件分成三份

vocabulary_path = path + 'paraphrase_vocab.json'  # path where the used vocabularies for question and answers are saved to
word_vectors_path = path + 'word_vectors.npy'

# train_data_path = path + 'sentences_train.json'
# dev_data_path = path + 'sentences_val.json'
# test_data_path = path + 'sentences_test.json'

path_data = './data_sentences/'
train_data_path = path_data + 'sentences_train.json'
dev_data_path = path_data + 'sentences_val.json'
test_data_path = path_data + 'sentences_test.json'
classes_path = path + "object_class_list.txt"
glove_index = '../data/dictionary.pkl'
embedding_path = '../data/glove6b_init_300d.npy'
checkpoint_path = './checkpoints/'
min_word_freq = 3
max_q_length = 666  # question_length = min(max_q_length, max_length_in_dataset)
max_a_length = 3
batch_size = 200
data_workers = 8
normalize_box = True
seed = 2021