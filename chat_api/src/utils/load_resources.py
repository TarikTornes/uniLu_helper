from dynaconf import Dynaconf
import pickle

def load_configs():
    return Dynaconf(settings_file=["../conf/configs.toml"])

def load_data():
    with open("../data/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    with open("../data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return chunks, embeddings
