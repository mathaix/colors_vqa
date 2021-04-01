from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
from skimage import io, transform
import csv
import os
import pandas as pd
import matplotlib as plt
from nltk.tokenize import word_tokenize
import nltk
from utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL


class ColorsVqaDataSet(Dataset):
    """
    Load colors dataset
    """

    def __init__(self, datadir, datafile):
        self.image_pattern = "image_%s.png"
        self.image_dir = os.path.join(datadir, "images")
        self.data = []

        datafilepath = os.path.join(datadir, datafile)
        self.data_frame = pd.read_csv(datafilepath)
        self._transform = transforms.Compose([
                                        transforms.ToTensor(),
                                    ])

        answer_text = self.data_frame["answer"].tolist()
        question_text = self.data_frame["question"].tolist()

        self.answer_train = [self._tokenize_sentence(s) for s in answer_text]
        self.question_train = [self._tokenize_sentence(s) for s in question_text]
        vocab_train =  self.answer_train + self.question_train
        self.vocab = sorted({w for toks in vocab_train for w in toks})
        self.vocab += [UNK_SYMBOL]
        self.vocab_size = len(self.vocab)

        self.word2index = dict(zip(self.vocab, range(self.vocab_size)))
        self.index2word = dict(zip(range(self.vocab_size), self.vocab))
        self.start_index = self.vocab.index(START_SYMBOL)
        self.end_index = self.vocab.index(END_SYMBOL)
        self.unk_index = self.vocab.index(UNK_SYMBOL)
        self.build_dataset()

    def _tokenize_sentence(self, s):
        words = word_tokenize(s)
        words=[word.lower() for word in words if word.isalpha()]
        return [START_SYMBOL] + words + [END_SYMBOL]

    def get_vocab(self):
        return self.vocab

    @staticmethod
    def collate_fn(batch):
        """
        Function for creating batches.

        Parameter
        ---------
        batch : tuple of length 3
            Contains the `color_seqs`, `word_seqs`, and `ex_lengths`,
            all as lists or similar Python iterables. The function
            turns them into Tensors.

        Returns
        -------
        image : torch.FloatTensor.
             The shape is `(m, n, p)` where `m` is the batch_size,
             `n` is the number of colors in each context, and `p` is
             the color dimensionality.

        answer_seqs : torch.LongTensor
            This is a padded sequence, dimension (m, k), where `m` is
            the batch_size and `k` is the length of the longest sequence
            in the batch.

        ex_lengths : torch.LongTensor
            The true lengths of each sequence in `word_seqs. This will
            have shape `(m, )`, where `m` is the batch_size.

        answer_targets :  torch.LongTensor
            This is a padded sequence, dimension (m, k-1), where `m` is
            the batch_size and `k` is the length of the longest sequence
            in the batch. The targets match `word_seqs` except we drop the
            first symbol, as it is always START_SYMBOL. When the loss is
            calculated, we compare this sequence to `word_seqs` excluding
            the final character, which is always the END_SYMBOL. The result
            is that each timestep t is trained to predict the symbol
            at t+1.

        """
        images, word_seqs, ex_lengths, question_seqs, question_lens = zip(*batch)

        word_seqs = [torch.LongTensor(seq) for seq in word_seqs]
        ex_lengths = torch.LongTensor(ex_lengths)

        question_seqs = [torch.LongTensor(seq) for seq in question_seqs]
        question_lengths = torch.LongTensor(question_lens)
        # Targets as next-word predictions:
        targets = [x[1:, ] for x in word_seqs]
        # Padding
        word_seqs = torch.nn.utils.rnn.pad_sequence(
            word_seqs, batch_first=True)
        
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True)
        
        question_seqs = torch.nn.utils.rnn.pad_sequence(
            question_seqs, batch_first=True)

        images_tensor = torch.stack(images)
        
        return images_tensor, word_seqs, ex_lengths, targets, question_seqs, question_lengths

    def build_dataset(self):
        self.answer_seqs = [[self.word2index.get(w, self.unk_index) for w in seq]
                     for seq in self.answer_train]
        self.answer_lengths = [len(seq) for seq in self.answer_seqs]

        self.question_seqs = [[self.word2index.get(w, self.unk_index) for w in seq]
                     for seq in self.question_train]
        self.question_lengths = [len(seq) for seq in self.question_seqs]

   
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]  
        img_name =  os.path.join(self.image_dir, self.image_pattern % (row['image_id']))
        image = io.imread(img_name)
        image = self._transform(image)

        return  image , self.answer_seqs[idx], self.answer_lengths[idx], self.question_seqs[idx], self.question_lengths[idx]

