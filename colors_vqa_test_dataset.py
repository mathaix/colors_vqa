from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import skimage 
import io
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from nltk.tokenize import word_tokenize
import nltk
from utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL
import colorsys
from skimage.transform import rescale, resize, downscale_local_mean



class ColorsVqaTestDataSet(Dataset):
    """
    Creates a simulated VQADataset from color_sequences, word_sequences and question_sequences
    """

    def __init__(self, color_seqs, word_seqs, word_len, question_seq, question_length, vocab):
        self.data = []

        self._transform = transforms.Compose([
                                        transforms.Scale((100,300)),
                                        transforms.ToTensor(),
                                    ])

        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        self.word2index = dict(zip(self.vocab, range(self.vocab_size)))
        self.index2word = dict(zip(range(self.vocab_size), self.vocab))
        self.start_index = self.vocab.index(START_SYMBOL)
        self.end_index = self.vocab.index(END_SYMBOL)
        self.unk_index = self.vocab.index(UNK_SYMBOL)
        self.answer_seqs = word_seqs
        self.answer_lengths = word_len
        self.question_seqs = question_seq
        self.question_lengths = question_length
        self.color_seqs = color_seqs


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
        images, word_seqs, ex_lengths, question_seqs, question_lengths = zip(*batch)

        word_seqs = [torch.LongTensor(seq) for seq in word_seqs]
        ex_lengths = torch.LongTensor(ex_lengths)
        # Targets as next-word predictions:
        targets = [x[1:, ] for x in word_seqs]
        # Padding
        word_seqs = torch.nn.utils.rnn.pad_sequence(
            word_seqs, batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True)

        question_seqs = [torch.LongTensor(seq) for seq in question_seqs]
        question_seqs = torch.nn.utils.rnn.pad_sequence(
            question_seqs, batch_first=True)

        question_lengths = torch.LongTensor(question_lengths)

        images_tensor = torch.stack(images)
        return images_tensor, word_seqs, ex_lengths, targets, question_seqs, question_lengths


    def __len__(self):
        return len(self.color_seqs)

    def __getitem__(self, idx):
        colors = self.color_seqs[idx]
        image = self.image(colors)
        image = self._transform(image)

        return  image , self.answer_seqs[idx], self.answer_lengths[idx], self.question_seqs[idx], self.question_lengths[idx]

        
    def image(self, colors):
        border="black"
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3, 1))
        rgb = [colorsys.hls_to_rgb(*c) for c in colors]
        target_index = 2
        for i, c in enumerate(rgb):
                if i == target_index:
                    ec = border
                else:
                    ec = c
                patch = mpatch.Rectangle((0, 0), 1, 1, color=c, ec=ec, lw=8)
                axes[i].add_patch(patch)
                axes[i].axis('off')

        #return self._get_img_from_fig(fig)
        img_name =  os.path.join("data", "temp.png")
        plt.savefig(img_name)
        plt.close()

        
        image = Image.open(img_name) #skimage.io.imread(img_name)
        return image

        

    def _get_img_from_fig(self, fig, dpi=180):
        import numpy as np
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        print(img_arr)
        buf.close()

        return img_arr


