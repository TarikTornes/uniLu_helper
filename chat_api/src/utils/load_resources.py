import pickle, toml

def load_configs():
    with open('../conf/configs.toml') as f:
        config = toml.load(f)
    return config

def load_data():
    with open("../data/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    with open("../data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return chunks, embeddings
