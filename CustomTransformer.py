import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LayerNormalization, Embedding, Dropout, TimeDistributed, Flatten

class PositionalEncoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, rate, max_seq_len, **kargs):
        super().__init__(**kargs)
        self.d_model = d_model
        self.positional_encoded = self.positional_encoding(max_seq_len)

        self.dropout = Dropout(rate)
 
    def positional_encoding(self, max_seq_len):
        # PE(pos, 2i) = sin(pos/10000^{2i/d_model}), 
        # PE(pos, 2i+1) = cos(pos/10000^{2i/d_model})

        # it is difficult to itemize a tensor, e.g. [:, 0::2], we will use numpy for this one only)
        # it is also impossible numpy to deal with tensor object(e.g seq_len from a tensor), we need to explicitly offer seq_len from the beggining

        position = np.arange(max_seq_len)[..., np.newaxis]
        i = np.arange(self.d_model) // 2
        i = 1 / np.power(10000, 2*i/self.d_model)[np.newaxis, ...]
        positional_encoded = np.matmul(position, i)

        positional_encoded[:, 0::2] = np.sin(positional_encoded[:, 0::2])
        positional_encoded[:, 1::2] = np.cos(positional_encoded[:, 1::2])

        return tf.cast(positional_encoded, tf.float32)
  
    def call(self, input):
        embedded = input * tf.sqrt(tf.cast(self.d_model, tf.float32)) 
        out = embedded + self.positional_encoded[:tf.shape(input)[1], :]
        out = self.dropout(out)
        
        return out

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kargs):
        super().__init__(**kargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads; assert d_model % num_heads == 0

        self.query_dense, self.key_dense, self.value_dense = Dense(d_model), Dense(d_model), Dense(d_model)
        self.dense = Dense(d_model)
    
    def split_heads(self, x):
        x = tf.reshape(x, shape=(tf.shape(x)[0], -1, self.num_heads, self.depth)) # first split d_model into num_heads and depth
        x = tf.transpose(x, perm=[0,2,1,3]) # and then transpose
        return x

    def undo_split_heads(self, x):
        x = tf.transpose(x, perm=[0,2,1,3])
        x = tf.reshape(x, shape=(tf.shape(x)[0], -1, self.d_model))
        return x

    def scaled_dot_product_attention(self, q, k, v, mask):
        # softmax(QK^T/sqrt(d_k))·V
        qk = tf.matmul(q, k, transpose_b=True)
        d_k = tf.cast(tf.shape(k), tf.float32)[-1]
        logits = qk / tf.sqrt(d_k)

        if mask is not None:
            logits += mask * -1e9
        
        attention_weight = tf.nn.softmax(logits, axis=-1) # row wise softmax
        output = tf.matmul(attention_weight, v)

        return output
        
    def call(self, query, key, value, mask):
        query_weight, key_weight, value_weight = self.query_dense(query), self.key_dense(key), self.value_dense(value)
        query_splitted, key_splitted, value_splitted = self.split_heads(query_weight), self.split_heads(key_weight), self.split_heads(value_weight)
        out = self.scaled_dot_product_attention(query_splitted, key_splitted, value_splitted, mask)
        out = self.undo_split_heads(out)
        out = self.dense(out)

        return out
    
class PointwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_pff, d_model, **kargs):
        super().__init__(**kargs)
        self.dense1 = Dense(d_pff, activation="relu")
        self.dense2 = Dense(d_model)
    
    def call(self, input):
        out = self.dense1(input)
        out = self.dense2(out)
        return out

class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate, epsilon, d_pff, **kargs):
        super().__init__(**kargs)

        self.multihead_self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1, self.dropout2 = Dropout(rate), Dropout(rate)
        self.layer_norm1, self.layer_norm2 = LayerNormalization(epsilon=epsilon), LayerNormalization(epsilon=epsilon)
        self.pff = PointwiseFeedForward(d_pff, d_model)

    def call(self, embedded_input, padding_mask, training):
        attentioned_output = self.multihead_self_attention(embedded_input, embedded_input, embedded_input, padding_mask)
        attentioned_output = self.dropout1(attentioned_output, training=training)
        attentioned_output = self.layer_norm1(embedded_input + attentioned_output)

        pff_output = self.pff(attentioned_output)
        pff_output = self.dropout2(pff_output, training=training)
        pff_output = self.layer_norm2(attentioned_output + pff_output)
        
        return pff_output

class StackedEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, rate, epsilon, d_pff, **kargs):
        super().__init__(**kargs)
        self.encoder_li = [Encoder(d_model, num_heads, rate, epsilon, d_pff) for _ in range(num_layers)]
    
    def call(self, x, padding_mask, training):
        for encoder in self.encoder_li:
            x = encoder(x, padding_mask, training)

        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate, epsilon, d_pff, **kargs):
        super().__init__(**kargs)
        self.multihead_self_attention = MultiHeadAttention(d_model, num_heads)
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.pff = PointwiseFeedForward(d_pff, d_model)
        
        self.dropout1, self.dropout2, self.dropout3 = Dropout(rate), Dropout(rate), Dropout(rate)
        self.layer_norm1, self.layer_norm2, self.layer_norm3 = LayerNormalization(epsilon=epsilon), LayerNormalization(epsilon=epsilon), LayerNormalization(epsilon=epsilon)
    
    def call(self, embedded_input, lookahead_mask, training, enc_output, padding_mask):
        attentioned_output1 = self.multihead_self_attention(embedded_input, embedded_input, embedded_input, lookahead_mask) # teacher_force
        attentioned_output1 = self.dropout1(attentioned_output1, training=training)
        attentioned_output1 = self.layer_norm1(attentioned_output1 + embedded_input)

        attentioned_output2 = self.multihead_attention(embedded_input, enc_output, enc_output, padding_mask)
        attentioned_output2 = self.dropout2(attentioned_output2, training=training)
        attentioned_output2 = self.layer_norm2(attentioned_output1 + attentioned_output2)

        pff_output = self.pff(attentioned_output2)
        pff_output = self.dropout3(pff_output, training=training)
        pff_output = self.layer_norm3(attentioned_output2 + pff_output)
        
        return pff_output

class StackedDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, rate, epsilon, d_pff, **kargs):
        super().__init__(**kargs)
        self.decoder_li = [Decoder(d_model, num_heads, rate, epsilon, d_pff) for _ in range(num_layers)]
    
    def call(self, x, lookahead_mask, training, enc_output, padding_mask):
        for decoder in self.decoder_li:
            x = decoder(x, lookahead_mask, training, enc_output, padding_mask)
        return x

class CustomTransformer(tf.keras.layers.Layer):
    def __init__(self, movie_name_vocab_size, d_model, rate, max_seq_len, num_heads, num_layers, epsilon, d_pff, data_dict, **kargs):
        super().__init__(**kargs)
        # for encoders
        self.movie_name_embedding = Embedding(movie_name_vocab_size, d_model)
        
        self.site_name_embedding = Embedding(data_dict["site_name_data"].nunique()[0]+1, d_model) # +1 for unknown
        self.art_mov_yn_embedding = Embedding(data_dict["art_mov_yn_data"].nunique()[0]+1, d_model)
        self.mov_klj_yn_embedding = Embedding(data_dict["mov_klj_yn_data"].nunique()[0]+1, d_model)
        self.nations_embedding = Embedding(data_dict["nations_data"].nunique()[0]+1, d_model)
        self.genres1_embedding = Embedding(data_dict["genres1_data"].nunique()[0]+1, d_model)
        self.genres2_embedding = Embedding(data_dict["genres2_data"].nunique()[0]+1, d_model)
        self.director1_embedding = Embedding(data_dict["director1_data"].nunique()[0]+1, d_model)
        self.director2_embedding = Embedding(data_dict["director2_data"].nunique()[0]+1, d_model)
        self.actor1_embedding = Embedding(data_dict["actor1_data"].nunique()[0]+1, d_model)
        self.actor2_embedding = Embedding(data_dict["actor2_data"].nunique()[0]+1, d_model)
        self.actor3_embedding = Embedding(data_dict["actor3_data"].nunique()[0]+1, d_model)
        self.rating_embedding = Embedding(data_dict["rating_data"].nunique()[0]+1, d_model)
        self.mov_knd_nm_embedding = Embedding(data_dict["mov_knd_nm_data"].nunique()[0]+1, d_model)
        
        self.pos_enc = PositionalEncoder(movie_name_vocab_size, d_model, rate, max_seq_len)
        self.stacked_encoder = StackedEncoder(num_layers, d_model, num_heads, rate, epsilon, d_pff)

        # for decoders
        self.dec_mov_tkt_dense = TimeDistributed(Dense(d_model))
        self.dec_daynames_embedding = Embedding(len({i for sub in data_dict["daynames_data"] for i in sub})+1, d_model)
        self.dec_holiday_embedding = Embedding(len({i for sub in data_dict["holiday_data"] for i in sub})+1, d_model)

        self.decoder_dense = TimeDistributed(Dense(d_model, activation="relu"))
        self.stacked_decoder = StackedDecoder(num_layers, d_model, num_heads, rate, epsilon, d_pff)

        # final dense
        self.dense = Dense(1, activation="sigmoid")

    def create_padding_mask(self, x):
        padding_mask = tf.cast(tf.equal(0., x), tf.float32) # each row is a sentence
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :] # split in order to apply to look ahead mask as well as scaled dot product attention
        return padding_mask

    def create_lookahead_mask(self, x):
        size = tf.shape(x)[-1]
        lookahead_mask = 1 - tf.linalg.band_part(tf.ones(shape=(size, size)), -1, 0) # Look ahead mask is applied to K·Q (seq_len, seq_len). Note the shape of the mask
        # padding_mask = self.create_padding_mask(x)
        # return tf.maximum(lookahead_mask, padding_mask)
        return lookahead_mask

    def call(self, movie_name_input, site_name_input, art_mov_yn_input, mov_klj_yn_input, nations_input, genres1_input, genres2_input, director1_input, director2_input, actor1_input, actor2_input, actor3_input, rating_input, mov_knd_nm_input, mov_tkt_input, daynames_input, holiday_input, training):
        # movie_name embedding
        movie_name_padding_mask = self.create_padding_mask(movie_name_input)
        embedded_movie_name = self.movie_name_embedding(movie_name_input)
        pos_encoded_movie_name = self.pos_enc(embedded_movie_name)
        enc_output = self.stacked_encoder(pos_encoded_movie_name, movie_name_padding_mask, training)

        # other info embedding
        embedded_site_name = self.site_name_embedding(site_name_input)
        embedded_art_mov_yn = self.art_mov_yn_embedding(art_mov_yn_input)
        embedded_mov_klj_yn = self.mov_klj_yn_embedding(mov_klj_yn_input)
        embedded_nations = self.nations_embedding(nations_input)
        embedded_genres1 = self.genres1_embedding(genres1_input)
        embedded_genres2 = self.genres2_embedding(genres2_input)
        embedded_director1 = self.director1_embedding(director1_input)
        embedded_director2 = self.director2_embedding(director2_input)
        embedded_actor1 = self.actor1_embedding(actor1_input)
        embedded_actor2 = self.actor2_embedding(actor2_input)
        embedded_actor3 = self.actor3_embedding(actor3_input)
        embedded_rating = self.rating_embedding(rating_input)
        embedded_mov_knd_nm = self.mov_knd_nm_embedding(mov_knd_nm_input)

        # enc_concat
        enc_concat = tf.keras.layers.concatenate([enc_output, embedded_site_name, embedded_art_mov_yn, embedded_mov_klj_yn, embedded_nations, embedded_genres1, embedded_genres2, embedded_director1, embedded_director2, embedded_director2, embedded_actor1, embedded_actor2, embedded_actor3, embedded_rating, embedded_mov_knd_nm], axis=1)
        
        # decoder setting
        dec_padding_mask = movie_name_padding_mask
        zeros = tf.zeros(shape=(tf.shape(enc_concat)[0], 1, 1, 14), dtype=tf.float32)
        dec_padding_mask = tf.concat((dec_padding_mask, zeros), axis=-1)

        dec_lookahead_mask = self.create_lookahead_mask(mov_tkt_input)

        # teacher_concat
        # teacher_concat = tf.keras.layers.concatenate([mov_tkt_input[..., tf.newaxis], daynames_input[..., tf.newaxis], holiday_input[..., tf.newaxis]])
        densed_mov_tkt = self.dec_mov_tkt_dense(mov_tkt_input[..., tf.newaxis])
        densed_daynames = self.dec_daynames_embedding(daynames_input)
        densed_holiday = self.dec_holiday_embedding(holiday_input)
        teacher_concat = tf.keras.layers.concatenate([densed_mov_tkt, densed_daynames, densed_holiday])
        
        teacher_concat = self.decoder_dense(teacher_concat)
        pos_encoded_teacher = self.pos_enc(teacher_concat)
        dec_output = self.stacked_decoder(pos_encoded_teacher, dec_lookahead_mask, training, enc_concat, dec_padding_mask)
        dec_output = self.dense(dec_output)

        return dec_output