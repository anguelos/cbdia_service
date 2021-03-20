import torch
import numpy as np
import time
import pickle
import glob

from typing import Tuple, List
from .hinttypes import t_filename, t_indexreply_ind, t_rect, t_optional_tensor, t_image, img_to_array, t_ctx
from .embedder import AbstractStringImageEmbedder


class AbstractIndex(object):
    @property
    def nb_embeddings(self):
        raise NotImplemented

    @property
    def embedding_size(self):
        raise NotImplemented

    @property
    def nb_documents(self):
        raise NotImplemented

    def search(self, query_embedding: np.array, ctx_docnames: t_ctx, max_occurence_per_document: int)->List:
        raise NotImplemented

    def save(self, path):
        raise NotImplemented

    @classmethod
    def load(cls, path):
        raise NotImplemented


class NumpyIndex(AbstractIndex):
    @property
    def nb_embeddings(self):
        return self.embeddings.shape[0]

    @property
    def embedding_size(self):
        return self.embeddings.shape[1]

    @property
    def nb_documents(self):
        return self.docnames.shape[0]

    def __init__(self, nb_embeddings: int, embedding_size: int, nb_documents: int, metric:str, embedding_dtype=np.float16):
        assert metric in ["euclidean"]
        self.metric = metric
        self.docnames = np.empty(nb_documents, dtype=str)
        self.idx = np.arange(nb_embeddings, dtype=int)
        self.embeddings = np.empty([nb_embeddings, embedding_size],dtype=embedding_dtype)
        self.doccodes = np.empty(nb_embeddings, dtype=int)
        self.pagecodes = np.empty(nb_embeddings, dtype=int)
        self.left = np.empty(nb_embeddings, dtype=int)
        self.top = np.empty(nb_embeddings, dtype=int)
        self.right = np.empty(nb_embeddings, dtype=int)
        self.bottom = np.empty(nb_embeddings, dtype=int)
        self.image_widths = np.empty(nb_embeddings, dtype=int)
        self.image_heights = np.empty(nb_embeddings, dtype=int)

    def _reset_nb_embeddings(self, nb_embeddings: int)->None:
        self.idx = np.arange(nb_embeddings, dtype=int)
        self.embeddings = np.empty([nb_embeddings, self.embedding_size])
        self.doccodes = np.empty(nb_embeddings, dtype=int)
        self.pagecodes = np.empty(nb_embeddings, dtype=int)
        self.left = np.empty(nb_embeddings, dtype=int)
        self.top = np.empty(nb_embeddings, dtype=int)
        self.right = np.empty(nb_embeddings, dtype=int)
        self.bottom = np.empty(nb_embeddings, dtype=int)
        self.image_widths = np.empty(nb_embeddings, dtype=int)
        self.image_heights = np.empty(nb_embeddings, dtype=int)

    def _reset_embedding_size(self, embedding_size: int)->None:
        self.embeddings = np.empty([self.nb_embeddings, embedding_size])

    def _reset_num_documents(self, nb_documents: int)->None:
        self.docnames = np.empty(nb_documents, dtype=str)

    def _idx_to_response(self, idx=np.array)->List:
        docnames = self.docnames[self.doccodes[idx]].tolist()
        pagecodes = self.pagecodes[idx].tolist()
        left = self.left[idx].tolist()
        top = self.top[idx].tolist()
        right = self.right[idx].tolist()
        bottom = self.right[idx].tolist()
        width = self.width[idx].tolist()
        height = self.height[idx].tolist()
        return zip(docnames, pagecodes, left, top, right, bottom, width, height)

    def _context_to_idx(self, ctx_docnames: t_ctx)->np.array:
        if len(ctx_docnames) == 0:
            idx = np.ones(self.nb_embeddings, dtype=np.bool)
        else:
            idx = np.zeros(self.nb_embeddings, dtype=np.bool)
            for docname in ctx_docnames:
                doccodes = np.nonzero(self.docnames == docname)[0]
                assert len(doccodes) == 1
                idx = idx | self.doccodes == doccodes[0]
        return idx

    def _retrieve_euclidean(self, query_embedding: np.array, ctx_docnames: t_ctx)->Tuple[np.array, np.array]:
        ctx_idx = self.context_to_idx(ctx_docnames)
        embeddings = self.embeddings[ctx_idx, :]
        reversed_ctx_idx = self.idx[ctx_idx]
        subtracted = embeddings - query_embedding
        similarity = (subtracted * subtracted).sum(axis=1) ** .5
        sorted_ctx_idx = np.argsort(similarity)
        response_idx = reversed_ctx_idx[sorted_ctx_idx]
        return response_idx, similarity

    def _generate_perdoc_occurence_limit(self, response_idx:np.array, max_responces_perdoc:int)->np.array:
        if max_responces_perdoc <=0 :
            return np.ones(response_idx.shape, dtype=np.bool)
        else:
            response_doccodes = self.doccodes[response_idx]
            occurences = np.zeros([response_idx.shape[0], self.docnames.shape[0]])
            occurences[np.arange(response_idx.shape[0], dtype=int), response_doccodes] = 1
            occurence_count = (occurences.cumsum(axis=0) * occurences).sum(axis=1)
            return occurence_count <= max_responces_perdoc

    def search(self, query_embedding: np.array, ctx_docnames: t_ctx, max_occurence_per_document: int)->List:
        if self.metric=="euclidean":
            responce_idx, similarity = self._retrieve_euclidean(query_embedding, ctx_docnames)
        else:
            raise NotImplementedError(self.metric)
        filter_by_doc_occurences = self._generate_perdoc_occurence_limit(responce_idx, max_occurence_per_document)
        responce_idx, similarity = responce_idx[filter_by_doc_occurences], similarity[filter_by_doc_occurences]
        return self._idx_to_response(responce_idx)

    def save(self, path):
        with open(path,"wb") as fd:
            data = {
                "metric": self.metric,
                "nb_embeddings": self.nb_embeddings,
                "embedding_size": self.embedding_size,
                "num_documents": self.nb_documents,
                "docnames":self.docnames,
                "embeddings": self.embeddings,
                "doccodes":self.doccodes,
                "pagecodes":self.pagecodes,
                "left":self.left,
                "top":self.top,
                "right":self.right,
                "bottom":self.bottom,
                "image_widths":self.image_widths,
                "image_heights":self.image_heights
            }
            pickle.dump(data, fd)

    @classmethod
    def load(cls, path):
        data = pickle.load(open(path,"rb"))
        result = cls(nb_embeddings=data["nb_embeddings"], embedding_size=data["embedding_size"], nb_documents=data["nb_documents"])
        del data["nb_embeddings"]
        del data["embedding_size"]
        del data["nb_documents"]
        result.__dict__.update(data)
        result.idx = np.arange(result.nb_embeddings, dtype=int)
        return result

    def set_random_data(self, pages_per_doc, page_width, page_height, min_word_height, max_word_height, min_word_width, max_word_width, doc_glob, doc_root):
        print("Creating random data ... ", end="")
        t = time.time()
        names = sorted([path[len(doc_root):] for path in glob.glob(doc_glob)])
        for n in range(min(len(names), self.nb_documents)):
            self.docnames[n] = names[n]
        self.embeddings[:, :self.embedding_size//2] = np.random.rand(self.nb_embeddings, self.embedding_size//2)
        self.embeddings[:, self.embedding_size // 2:] = np.random.rand(self.nb_embeddings, self.embedding_size - self.embedding_size // 2)
        self.left = np.random.randint(0, page_width, self.nb_embeddings)
        self.top = np.random.randint(0, page_height, self.nb_embeddings)
        self.right = self.left + np.random.randint(min_word_width, max_word_width,
                                                                          self.nb_embeddings)
        self.bottom = self.top + np.random.randint(min_word_height,
                                                                          max_word_height, self.nb_embeddings)
        self.doccodes = np.random.randint(0, min(len(names), self.nb_documents), self.nb_embeddings)
        self.pagecodes = np.random.randint(0, pages_per_doc, self.nb_embeddings) * 10
        print("Done!")