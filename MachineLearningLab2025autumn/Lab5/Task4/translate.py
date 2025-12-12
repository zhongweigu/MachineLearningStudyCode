import os
import pickle
import numpy as np
import nltk
from typing import Dict, Tuple, List
from nltk.corpus import twitter_samples

# Only use provided utils (per assignment requirements)
from utils import process_tweet, cosine_similarity


def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # cast to float32 for consistency/perf
    for k, v in list(data.items()):
        if not isinstance(v, np.ndarray):
            data[k] = np.array(v, dtype=np.float32)
        else:
            data[k] = v.astype(np.float32)
    return data


def get_document_embedding(tweet: str, en_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Sum of word embeddings for tokens in the tweet (BOW document embedding).
    Unknown words are skipped. Returns zero vector if no token has an embedding.
    """
    vec_dim = len(next(iter(en_embeddings.values())))
    doc_vec = np.zeros((vec_dim,), dtype=np.float32)
    tokens = process_tweet(tweet)
    for w in tokens:
        if w in en_embeddings:
            doc_vec += en_embeddings[w]
    return doc_vec


def get_document_vecs(all_docs: List[str], en_embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Build matrix of tweet embeddings and index->tweet mapping.
    Returns:
      - document_vec_matrix: shape (num_docs, vec_dim)
      - ind2Tweet: dict mapping row index -> original tweet string
    """
    vec_dim = len(next(iter(en_embeddings.values())))
    m = len(all_docs)
    mat = np.zeros((m, vec_dim), dtype=np.float32)
    ind2Tweet = {}
    for i, tw in enumerate(all_docs):
        mat[i, :] = get_document_embedding(tw, en_embeddings)
        ind2Tweet[i] = tw
    return mat, ind2Tweet


def hash_value_of_vector(v: np.ndarray, planes: np.ndarray) -> int:
    """Create a hash for a vector using a set of hyperplanes.
    v: (N_DIMS,) or (1, N_DIMS)
    planes: (N_DIMS, N_PLANES)
    hash = sum(2^i * h_i), where h_i is 1 if dot(v, plane_i) >= 0 else 0
    """
    if v.ndim == 2 and v.shape[0] == 1:
        v = v.reshape(-1)
    projections = np.dot(v, planes)  # (N_PLANES,)
    bits = (projections >= 0).astype(int)
    # compute integer hash
    h = 0
    for i, b in enumerate(bits):
        if b:
            h += (1 << i)
    return int(h)


def make_hash_table(vecs: np.ndarray, planes: np.ndarray):
    """
    Build a single-universe hash table and id table.
      - vecs: (N_VECS, N_DIMS)
      - planes: (N_DIMS, N_PLANES)
    Returns (hash_table, id_table)
      hash_table: dict[hash] -> list of vectors (as numpy arrays)
      id_table: dict[hash] -> list of vector indices (doc ids)
    """
    hash_table: Dict[int, List[np.ndarray]] = {}
    id_table: Dict[int, List[int]] = {}
    for idx in range(vecs.shape[0]):
        v = vecs[idx, :]
        h = hash_value_of_vector(v, planes)
        if h not in hash_table:
            hash_table[h] = []
            id_table[h] = []
        hash_table[h].append(v)
        id_table[h].append(idx)
    return hash_table, id_table


# Globals built once and reused by approximate_knn
hash_tables: List[Dict[int, List[np.ndarray]]] = []
id_tables: List[Dict[int, List[int]]] = []
planes_l: List[np.ndarray] = []
document_vecs: np.ndarray
ind2Tweet: Dict[int, str]


def approximate_knn(doc_id: int, v: np.ndarray, planes_l_in: List[np.ndarray], k: int = 1, num_universes_to_use: int = 25) -> List[int]:
    """Search for k-NN using LSH buckets across multiple universes.
    Collect candidate ids from matching buckets, then rank by cosine similarity.
    """
    assert num_universes_to_use <= len(planes_l_in)
    candidates: List[int] = []
    seen = set()
    for uni in range(num_universes_to_use):
        planes = planes_l_in[uni]
        h = hash_value_of_vector(v, planes)
        ids_here = id_tables[uni].get(h, [])
        for cid in ids_here:
            if cid != doc_id and cid not in seen:
                seen.add(cid)
                candidates.append(cid)
    # If no candidates found, relax by returning empty list
    if not candidates:
        return []
    # Rank candidates by cosine similarity
    scored = []
    for cid in candidates:
        sim = cosine_similarity(v, document_vecs[cid, :])
        scored.append((cid, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in scored[:k]]
    return top_ids


def main():
    # Ensure NLTK resources (prefer local tmp2 cache)
    try:
        import os
        base = os.path.dirname(os.path.dirname(__file__))  # .../Lab5
        tmp2_path = os.path.join(base, 'tmp2')
        import nltk
        if os.path.isdir(tmp2_path):
            if tmp2_path not in nltk.data.path:
                nltk.data.path.append(tmp2_path)
    except Exception:
        pass
    # Download if needed (no-op if already available)
    # nltk.download('stopwords')
    # nltk.download('twitter_samples')

    # Load embeddings subset (provided with assignment)
    en_path = os.path.join(os.path.dirname(__file__), 'en_embeddings.p')
    en_embeddings_subset = load_embeddings(en_path)

    # Load tweets
    pos = twitter_samples.strings('positive_tweets.json')
    neg = twitter_samples.strings('negative_tweets.json')
    all_tweets = pos + neg

    # Build document vectors
    global document_vecs, ind2Tweet
    document_vecs, ind2Tweet = get_document_vecs(all_tweets, en_embeddings_subset)
    print(f"length of dictionary {len(ind2Tweet)}")
    print(f"shape of document_vecs {document_vecs.shape}")

    # LSH setup
    N_VECS = document_vecs.shape[0]
    N_DIMS = document_vecs.shape[1]
    N_PLANES = 10
    N_UNIVERSES = 25
    global planes_l
    np.random.seed(0)
    planes_l = [np.random.normal(size=(N_DIMS, N_PLANES)).astype(np.float32) for _ in range(N_UNIVERSES)]

    # Quick test hash value function
    idx = 0
    planes = planes_l[idx]
    vec = np.random.rand(1, N_DIMS).astype(np.float32)
    hv = hash_value_of_vector(vec, planes)
    print(f"The hash value for a random vector with planes[{idx}] is {hv}")

    # Build all hash tables
    global hash_tables, id_tables
    hash_tables = []
    id_tables = []
    for universe_id in range(N_UNIVERSES):
        print('working on hash universe #:', universe_id)
        planes = planes_l[universe_id]
        hash_table, id_table = make_hash_table(document_vecs, planes)
        hash_tables.append(hash_table)
        id_tables.append(id_table)

    # Demo approximate KNN
    doc_id = 0
    doc_to_search = all_tweets[doc_id]
    vec_to_search = document_vecs[doc_id]

    neighbor_ids = approximate_knn(doc_id, vec_to_search, planes_l, k=3, num_universes_to_use=5)
    print(f"Nearest neighbors for document {doc_id}")
    print(f"Document contents: {doc_to_search}\n")
    for nid in neighbor_ids:
        print(f"Nearest neighbor at document id {nid}")
        print(f"document contents: {all_tweets[nid]}")


if __name__ == "__main__":
    main()
