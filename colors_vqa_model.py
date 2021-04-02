from torch.utils.data import DataLoader
from colors_vqa_dataset import ColorsVqaDataSet
from colors_vqa_test_dataset import ColorsVqaTestDataSet
from colors import ColorsCorpusReader
from vqa_decoder import VQADecoder
from image_encoder import ImageEncoder
from question_encoder import QuestionEncoder
import torch
import torch.nn as nn
from utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL
import torch.nn.functional as F 
import shutil
import os
import itertools
import numpy as np
from nltk.tokenize import word_tokenize
import nltk


checkpoint_dir = "./model"

def save_ckp(state, is_best=False, checkpoint_dir="model", best_model_dir="./model"):
    """
      Helper function to checkpoint and save a model during training
    """
    f_path = os.path.join(checkpoint_dir, "checkpoint2.pt")
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
      Helper function to load a saved model
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']



class VQAEncoderDecoder(nn.Module):
    """
    This the primary model, it is a VQA (Visual Question Answering model). It ties together an ImageEncoder (based on a CNN), 
    a Question Encoder (based on a GRU cell) and a Decoder based on a GRU
    """
    def __init__(self,
            vocab,
            embedding=None,
            embed_dim=50,
            hidden_dim=3520,
            freeze_embedding=False,
            device=None,
            **base_kwargs):

        super().__init__(**base_kwargs)
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.freeze_embedding = freeze_embedding
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        self.output_dim = self.vocab_size
        self.word2index = dict(zip(self.vocab, range(self.vocab_size)))
        self.index2word = dict(zip(range(self.vocab_size), self.vocab))
        self.start_index = self.vocab.index(START_SYMBOL)
        self.end_index = self.vocab.index(END_SYMBOL)
        self.unk_index = self.vocab.index(UNK_SYMBOL)
        self.params = ['hidden_dim', 'embed_dim', 'embedding', 'freeze_embedding']
        self.loss = nn.CrossEntropyLoss()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)


        self.encoder = ImageEncoder()
        self.question_encoder = QuestionEncoder(vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim,
            freeze_embedding=self.freeze_embedding)
        self.decoder = VQADecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim * 2,
            freeze_embedding=self.freeze_embedding)
    
    def build_dataset(self, color_seqs, word_seqs, question_seqs):
        """
        Create a VQA dataset from a list of color contexts and
        associated utterances.

        Parameters
        ----------
        color_seqs : list of lists of color representations
            We assume that each context has the same number of colors,
            each with the same shape.

        word_seqs : list of lists of utterances
            A tokenized list of words. This method uses `self.word2index`
            to turn this into a list of lists of indices.

        Returns
        -------
        ColorVQADataset

        """
        self.color_dim = len(color_seqs[0][0])
        word_seqs = [[self.word2index.get(w, self.unk_index) for w in seq]
                for seq in word_seqs]

        ex_lengths = [len(seq) for seq in word_seqs]
        question_seqs = [[self.word2index.get(w, self.unk_index) for w in seq]
                    for seq in question_seqs]
        question_lengths = [len(seq) for seq in question_seqs]
        return ColorsVqaTestDataSet(color_seqs, word_seqs, ex_lengths, 
                question_seqs, question_lengths,self.vocab)

    def predict_proba(self, color_seqs, word_seqs, question_seq, device=None):
        """
        Calculate the predicted probabilties of the sequences in
        `word_seqs` given the color contexts in `color_seqs`.

        Parameters
        ----------
        color_seqs : list of lists of lists of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.

        word_seqs : list of list of int
            Dimension m, the number of examples. The length of each
            sequence can vary.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.


        Returns
        -------
        list of lists of predicted probabilities. In other words,
        for each example, at each timestep, there is a probability
        distribution over the entire vocabulary.

        """
        device = self.device if device is None else torch.device(device)
        dataset = self.build_dataset(color_seqs, word_seqs, question_seq)

        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=len(word_seqs),
                shuffle=False,
                pin_memory=True,
                collate_fn=dataset.collate_fn)
  

        self.to(device)
        self.eval()
        softmax = nn.Softmax(dim=2)

        start_probs = np.zeros(self.vocab_size)
        start_probs[self.start_index] = 1.0

        all_probs = []

        with torch.no_grad():

            for batch_images, batch_words, batch_lens, targets, questions, question_length in dataloader:

                batch_images = batch_images.to(device)
                batch_words = batch_words.to(device)
                batch_lens = batch_lens.to(device)
                questions = questions.to(device)
                question_length = question_length.to(device)

                output, _ = self(
                    image=batch_images,
                    word_seqs=batch_words,
                    seq_lengths=batch_lens,
                    question_seqs =questions,
                    question_length = question_length
                )

                probs = softmax(output)
                probs = probs.cpu().numpy()
                probs = np.insert(probs, 0, start_probs, axis=1)
                all_probs += [p[: n] for p, n in zip(probs, batch_lens)]

        self.to(self.device)

        return all_probs

    def perplexities(self, color_seqs, word_seqs, question_seq, device=None):
        """
        Compute the perplexity of each sequence in `word_seqs`
        given `color_seqs`. For a sequence of conditional probabilities
        p1, p2, ..., pN, the perplexity is calculated as

        (p1 * p2 * ... * pN)**(-1/N)

        Parameters
        ----------
        color_seqs : list of lists of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.

        word_seqs : list of list of int
            Dimension m, the number of examples, and the length of
            each sequence can vary.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        list of float

        """
        probs = self.predict_proba(color_seqs, word_seqs, question_seq, device=device)
        scores = []
        for pred, seq in zip(probs, word_seqs):
            # Get the probabilities corresponding to the path `seq`:
            s = np.array([t[self.word2index.get(w, self.unk_index)]
                         for t, w in zip(pred, seq)])
            scores.append(s)
        perp = [np.prod(s)**(-1/len(s)) for s in scores]      
        return perp

    def listener_predict_one(self, context, seq, device=None):
        context = np.array(context)
        n_colors = len(context)

        # Get all possible context orders:
        indices = list(range(n_colors))
        orders = [list(x) for x in itertools.permutations(indices)]

        # All contexts as color sequences:
        contexts = [context[x] for x in orders]

        # Repeat the single utterance the needed number of times:
        seqs = [seq] * len(contexts)
        question_seqs = ["Describe the selected color"]  * len(contexts)


        question_seqs = [self._tokenize_sentence(s) for s in question_seqs]


        # All perplexities:
        perps = self.perplexities(contexts, seqs, question_seqs,device=device)


        # Ranking, using `order_indices` rather than colors and
        # index sequences to avoid sorting errors from some versions
        # of Python:
        order_indices = range(len(orders))
        ranking = sorted(zip(perps, order_indices))


        # Return the minimum perplexity, the chosen color, and the
        # index of the chosen color in the original context:
        min_perp, order_index = ranking[0]
        pred_color = contexts[order_index][-1]
        pred_index = orders[order_index][-1]
        return min_perp, pred_color, pred_index

    def listener_accuracy(self, color_seqs, word_seqs, device=None):
        """
        Compute the "listener accuracy" of the model for each example.
        For the ith example, this is defined as

        prediction = max_{c in C_i} P(word_seq[i] | c)

        where C_i is every possible permutation of the three colors in
        color_seqs[i]. We take the model's prediction to be correct
        if it chooses a c in which the target is in the privileged final
        position in the color sequence. (There are two such c's, since
        the distractors can be in two orders; we give full credit if one
        of these two c's is chosen.)

        Parameters
        ----------
        color_seqs : list of lists of list of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.

        word_seqs : list of list of int
            Dimension m, the number of examples, and the length of
            each sequence can vary.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        list of float

        """

        word_seqs = [self._tokenize_sentence(s) for s in word_seqs]

    
        correct = 0
        count = 0
        for color_seq, word_seq in zip(color_seqs, word_seqs):
            target_index = len(color_seq) - 1
            min_perp, pred, pred_index = self.listener_predict_one(
                color_seq, word_seq, device=device)
            count = count + 1
            correct += int(target_index == pred_index)
            print (count, correct)
        return correct / len(color_seqs)

    def _tokenize_sentence(self, s):
        words = word_tokenize(s)
        words=[word.lower() for word in words if word.isalpha()]
        return [START_SYMBOL] + words + [END_SYMBOL]

    def predict(self, color_images, question_seqs, question_length, max_length=20, device=None):
            """
            Predict new sequences based on the color contexts in
            `color_images`.

            Parameters
            ----------
            color_seqs : list of lists of lists of floats, or np.array
                Dimension (m, n, p) where m is the number of examples, n is
                the number of colors in each context, and p is the length
                of the color representations.

            max_length : int
                Length of the longest sequences to create.

            device: str or None
                Allows the user to temporarily change the device used
                during prediction. This is useful if predictions require a
                lot of memory and so are better done on the CPU. After
                prediction is done, the model is returned to `self.device`.

            Returns
            -------
            list of str

            """
            device = self.device if device is None else torch.device(device)

            color_images = color_images.to(device)
            self.decoder.to(device)
            self.decoder.eval()

            preds = []

            with torch.no_grad():
                # Get the hidden representations from the color contexts:
                image_hidden = self.encoder(color_images)
                _, question_hidden = self.question_encoder(
                    question_seqs=question_seqs, 
                    question_length=question_length, 
                    hidden=image_hidden)
                    
                hidden = torch.cat((image_hidden, question_hidden), 2)
                

                # Start with START_SYMBOL for all examples:
                decoder_input = [[self.start_index]] * len(color_images)
                decoder_input = torch.LongTensor(decoder_input)
                decoder_input = decoder_input.to(device)

                preds.append(decoder_input)

                # Now move through the remaiming timesteps using the
                # previous timestep to predict the next one:
                for i in range(1, max_length):

                    output, hidden = self.decoder(
                        decoder_input, 
                        seq_lengths=None, 
                        hidden=hidden)

                    # Always take the highest probability token to
                    # be the prediction:
                    p = output.argmax(2)
                    preds.append(p)
                    decoder_input = p

            # Convert all the predictions from indices to elements of
            # `self.vocab`:
            preds = torch.cat(preds, axis=1)
            preds = [self._convert_predictions(p) for p in preds]

            #self.model.to(self.device)

            return preds

    def _convert_predictions(self, pred):
        rep = []
        for i in pred:
            i = i.item()
            rep.append(self.index2word[i])
            if i == self.end_index:
                return rep
        return rep


    def forward(self, image, word_seqs, seq_lengths, question_seqs, question_length, hidden=None):
        """This is the core method for this module. It has a lot of
        arguments mainly to make it easy to create subclasses of this
        class that do interesting things without requring modifications
        to the `fit` method of `ContextualColorDescriber`.

        Parameters
        ----------
        color_seqs : torch.FloatTensor
            Dimension (m, p), where m is the number of examples,
            dimensionality of each image.

        word_seqs : torch.LongTensor
            Dimension (m, k), where m is the number of examples and k
            is the length of all the (padded) sequences in the batch.

        seq_lengths : torch.LongTensor or None
            The true lengths of the sequences in `word_seqs`. If this
            is None, then we are predicting new sequences, so we will
            continue predicting until we hit a maximum length or we
            generate STOP_SYMBOL.

        hidden : torch.FloatTensor or None
            The hidden representation for each of the m examples in this
            batch. If this is None, we are predicting new sequences
            and so the hidden representation is computed for each timestep
            during decoding.

        Returns
        -------
        output : torch.FloatTensor
            Dimension (m, k, c), where m is the number of examples, k
            is the length of the sequences in this batch, and c is the
            number of classes (the size of the vocabulary).

        hidden : torch.FloatTensor
            Dimension (m, h) where m is the number of examples and h is
            the dimensionality of the hidden representations of the model.
            This value is returned only when the model is in eval mode.

        """
        if hidden is None:
            image_hidden = self.encoder(image)
            _, question_hidden = self.question_encoder(
              question_seqs=question_seqs, question_length=question_length, hidden=image_hidden)
            hidden = torch.cat((image_hidden, question_hidden), 2)

            
        output, hidden = self.decoder(
            word_seqs, seq_lengths=seq_lengths, hidden=hidden)
        if self.training:
            return output
        else:
            return output, hidden

    def evaluate(self, color_seqs, word_seqs, device=None):
        """
        Full evaluation for the bake-off. Uses `listener_accuracy`
        and colors_corpus_bleu`.

        Parameters
        ----------
        color_seqs : list of lists of lists of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.

        word_seqs : list of lists of utterances
            A tokenized list of words.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        dict, {"listener_accuracy": float, 'corpus_bleu': float}

        """
        acc = self.listener_accuracy(color_seqs, word_seqs, device=device)
        ##bleu = self.corpus_bleu(color_seqs, word_seqs)
        return {"listener_accuracy": acc, 'corpus_bleu': None}


def main():
    train()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, batch_items in enumerate(dataloader):
        image, answer_seq, answer_len , target, question_seq, question_len = batch_items

        # Compute prediction and loss
        pred = model(image, answer_seq, answer_len,question_seq, question_len)
        loss = loss_fn(pred, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(batch_items)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def build_optimizer( model, eta = 0.001, l2_strength = 0):

    return torch.optim.Adam(model.parameters(), 
                           lr = eta,
                           weight_decay =  l2_strength
                           )


def train():
    batch_size = 64
    test_split = 0.7
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = ColorsVqaDataSet("data", "colors.csv")
    dataset_size = len(data)

    # Bakeoff dataset using to evaluate listner accuracy.
    COLORS_BAKEOFF_SRC_FILENAME = os.path.join(
    "data",  "cs224u-colors-bakeoff-data.csv")

    bakeoff_corpus = ColorsCorpusReader(COLORS_BAKEOFF_SRC_FILENAME)

    bakeoff_rawcols, bakeoff_texts = zip(*[
    [ex.colors, ex.contents] for ex in bakeoff_corpus.read()])
    
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size

    train_dataset, test_dataset = torch.utils.data.random_split(data,[train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn = data.collate_fn,
        shuffle=True)

    model = VQAEncoderDecoder(data.get_vocab())
    model.to(device)
    optimizer = build_optimizer(model)
    start_epoch = 0

    checkpoint_fpath = os.path.join(checkpoint_dir, "checkpoint2.pt")
    if os.path.isfile(checkpoint_fpath):
        model, optimizer, start_epoch = load_ckp(checkpoint_fpath, model, optimizer)   

    #print(model.evaluate(bakeoff_rawcols, bakeoff_texts, device=None)) 

    for epoch in range(start_epoch,epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
        save_ckp(checkpoint)
        
    print(model.evaluate(bakeoff_rawcols, bakeoff_texts, device=None))

    return model


if __name__ == "__main__":
   main()