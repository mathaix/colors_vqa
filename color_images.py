from colors import ColorsCorpusReader
import os


COLORS_SRC_FILENAME = os.path.join(
    "data", "colors", "filteredCorpus.csv")


COLORS_OUT_FILENAME = os.path.join(
    "data", "colors.csv")


corpus = ColorsCorpusReader(
    COLORS_SRC_FILENAME,
    word_count=None,
    normalize_colors=True)

import csv
i = 0

with open(COLORS_OUT_FILENAME, 'w', newline='') as csvfile:
    fieldnames = ['id', 'image_id', 'question', 'condition', 'answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for ex in corpus.read():
        ex.generate_data(writer, datadir="data/")