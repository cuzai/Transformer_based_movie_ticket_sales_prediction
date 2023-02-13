import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import sentencepiece as spm

class Tokenizer():
    def __init__(self, language):
        self.language = language

    def train(self, data): # Train sentencepeice
        # Save data for sentence piece to train
        with open(f'{self.language}_stpc.txt', 'w', encoding='utf8') as f:
            f.write('\n'.join(data))
        
        # Train spm
        spm.SentencePieceTrainer.Train(f"--input={self.language}_stpc.txt" +
                                        f" --model_prefix={self.language}" + 
                                        # " --vocab_size=8000" +
                                        # " --vocab_size=5555"+
                                        " --model_type=bpe" +
                                        " --max_sentence_length=999999"
                                        " --pad_id=0 --pad_piece=<PAD>" + # pad (0)
                                        " --unk_id=1 --unk_piece=<UNK>" + # unknown (1)
                                        " --bos_id=2 --bos_piece=<SOS>" + # begin of sequence (2)
                                        " --eos_id=3 --eos_piece=<EOS>" # end of sequence (3)
                                        )
        return self

    def tokenize(self, data):
        # Train sentencepiece only for the first time
        if not os.path.isfile("movie_stpc.txt"): 
            self.train(data)

        # Load
        sp = spm.SentencePieceProcessor()
        sp.Load(f"{self.language}.model")
        
        return sp

class Tokenize(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        self.tokenizer = Tokenizer("movie").tokenize(x["movie_name"])
        return self
    
    def transform(self, x, y=None):
        x = x.copy()
        x["movie_name"] = x["movie_name"].apply(lambda x: tuple(self.tokenizer.EncodeAsIds(x)))
        # x["movie_name"] = x["movie_name"].apply(lambda x: self.tokenizer.EncodeAsIds(x))
        return x
    
class LabelEncode(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        self.encoder_li = []
        self.encode_col = ["site_name", "art_mov_yn", "mov_klj_yn", "nations", "genres1", "genres2", "director1", "director2", "actor1", "actor2", "actor3", "rating", "mov_knd_nm", "daynames", "holiday"]
        for col in self.encode_col:
            encoder = LabelEncoder()
            encoder.fit(x[col])
            encoder = {val:n+1 for n, val in enumerate(encoder.classes_)}
            self.encoder_li.append(encoder)
        return self

    def transform(self, x, y=None):
        x = x.copy()
        for col, encoder in zip(self.encode_col, self.encoder_li):
            # x[col] = encoder.transform(x[col])
            x[col] = x[col].apply(lambda x: encoder[x] if x in encoder.keys() else 0)
        return x

class Pivot(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        x = x.copy()
        x = x.sort_values(["site_name", "movie_name", "date"])
        x = x.groupby(["site_name", "movie_name", "art_mov_yn", "mov_klj_yn", "nations", "genres1", "genres2", "director1", "director2", "actor1", "actor2", "actor3", "rating", "mov_knd_nm"], as_index=False)["mov_tkt", "daynames", "holiday"].agg(list)
        x["movie_name"] = x["movie_name"].map(np.array)
        return x