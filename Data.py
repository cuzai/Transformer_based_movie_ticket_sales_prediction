import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Data():
    def read_data(self, verbose):
        # Ticketing performance per theatre and movies
        df_tkt = pd.read_csv("../data/tkt_ratio_ver4_1109.csv", low_memory=False).drop("Unnamed: 0", axis=1)

        # Movie info
        df_mv_info = pd.read_csv("../data/movie_info.csv", low_memory=False).drop("Unnamed: 0", axis=1)[["movie_code", "movie_name", "open_dy", "art_mov_yn", "mov_klj_yn", "nations", "genres1", "genres2", "director1", "director2", "actor1", "actor2", "actor3", "rating", "mov_knd_nm"]]

        # Holiday_info
        df_holiday = pd.read_csv("../data/dt_holidays.csv", low_memory=False).drop("Unnamed: 0", axis=1)

        # Join
        df_prep_raw = df_tkt.merge(df_mv_info, on=["movie_code"], how="left")[["site_name", "site_code", "movie_name", "movie_code", "date"] + list(df_mv_info.drop(["movie_code", "movie_name"], axis=1).columns) + ["mov_tkt"]]
        df_prep_raw = df_prep_raw.merge(df_holiday, on=["date"], how="left")

        if verbose:
            print(f"df_tkt: {df_tkt.shape}, df_mv_info: {df_mv_info.shape}, df_holiday: {df_holiday.shape}")
            print(f"join result: {df_prep_raw.shape}")
        
        return df_prep_raw
    
    def preprocess(self, data, verbose):
        # Deal zero performance
        data.loc[data["mov_tkt"] == 0, "mov_tkt"] = 1e-9

        # Maximum performance should be greater than 300
        data["max_tkt"] = data.groupby(["site_name", "movie_name"],)["mov_tkt"].transform("max")
        data = data[data["max_tkt"] >= 300]


        # Missing values to "missing"
        for col in ["open_dy", "nations", "genres1", "genres2", "director1", "director2", "actor1", "actor2", "actor3"]:
            data[col] = data[col].fillna("missing")
            data.isna().sum().sum()
        
        # Open day and sceen start day should be the same
        data["min_date"] = data.groupby(["site_name", "movie_name"])["date"].transform("min")
        data = data[data["open_dy"] == data["min_date"]].drop(["min_date"], axis=1)

        # Scale
        max_val = data["mov_tkt"].max()
        data["mov_tkt"] = data["mov_tkt"] / max_val

        # Drop useless
        data = data.drop(["site_code", "movie_code"], axis=1)

        return data

    def get_buffer_size(self, num): # Get data.shape[0] and ceil for sufficient buffer size.
        # Make it as a string
        num = str(num) 
        first_digit = num[0]

        # Make ceil array
        arr = np.zeros(shape= (len(num),), dtype=np.int16)
        arr[0] = int(first_digit) + 1

        # Return it to a number again
        arr = int("".join([str(i) for i in arr]))
        
        return arr
    
    def get_tf_dataset(self, data, batch_size):
        # Define encoder input
        movie_name_data = pad_sequences(data["movie_name"], padding="post", dtype=np.float32)
        site_name_data = data[["site_name"]].astype(np.float32)
        art_mov_yn_data = data[["art_mov_yn"]].astype(np.float32)
        mov_klj_yn_data = data[["mov_klj_yn"]].astype(np.float32)
        nations_data = data[["nations"]].astype(np.float32)
        genres1_data = data[["genres1"]].astype(np.float32)
        genres2_data = data[["genres2"]].astype(np.float32)
        director1_data = data[["director1"]].astype(np.float32)
        director2_data = data[["director2"]].astype(np.float32)
        actor1_data = data[["actor1"]].astype(np.float32)
        actor2_data = data[["actor2"]].astype(np.float32)
        actor3_data = data[["actor3"]].astype(np.float32)
        rating_data = data[["rating"]].astype(np.float32)
        mov_knd_nm_data = data[["mov_knd_nm"]].astype(np.float32)

        # Define decoder input
        mov_tkt_data = pad_sequences(data["mov_tkt"].to_numpy(), padding="post", dtype=np.float32)[:, :30] # Get only 30 days of data
        daynames_data = pad_sequences(data["daynames"].to_numpy(), padding="post", dtype=np.float32)[:, :30]
        holiday_data = pad_sequences(data["holiday"].to_numpy(), padding="post", dtype=np.float32)[:, :30]

        # Define dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            (
            # encoder input
            movie_name_data,
            site_name_data,
            art_mov_yn_data,
            mov_klj_yn_data,
            nations_data,
            genres1_data,
            genres2_data,
            director1_data,
            director2_data,
            actor1_data,
            actor2_data,
            actor3_data,
            rating_data,
            mov_knd_nm_data,

            # decoder input
            mov_tkt_data[:, :-1],
            daynames_data[:, :-1],
            holiday_data[:, :-1],
            ),

            (mov_tkt_data[:, 1:])
        ))

        buffer_size = self.get_buffer_size(mov_tkt_data.shape[0])
        dataset = dataset.cache().shuffle(buffer_size=buffer_size).batch(batch_size=batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, {"mov_tkt_data":mov_tkt_data, "site_name_data":site_name_data, "art_mov_yn_data":art_mov_yn_data, "mov_klj_yn_data":mov_klj_yn_data, "nations_data":nations_data, "genres1_data":genres1_data, "genres2_data":genres2_data, "director1_data":director1_data, "director2_data":director2_data, "actor1_data":actor1_data, "actor2_data":actor2_data, "actor3_data":actor3_data, "rating_data":rating_data, "mov_knd_nm_data":mov_knd_nm_data, "daynames_data":daynames_data, "holiday_data":holiday_data}