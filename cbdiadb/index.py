import torch
import numpy as np
import time
import pickle
import glob
from collections import defaultdict

from typing import Tuple, List
from .hinttypes import t_filename, t_indexreply_ind, t_rect, t_optional_tensor, t_image, img_to_array, t_ctx

import os


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

    def get_docname_reverse_index(self):
        return {name: n for n, name in enumerate(sorted(self.docnames.tolist()))}

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
        self.docnames = np.empty(nb_documents, dtype=object)
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
        self.docnames = np.empty(nb_documents, dtype=object)

    def _idx_to_response(self, idx=np.array)->List:
        docnames = self.docnames[self.doccodes[idx]].tolist()
        pagecodes = self.pagecodes[idx].tolist()
        left = self.left[idx].tolist()
        top = self.top[idx].tolist()
        right = self.right[idx].tolist()
        bottom = self.right[idx].tolist()
        width = self.image_widths[idx].tolist()
        height = self.image_heights[idx].tolist()
        res = list(zip(docnames, pagecodes, left, top, right, bottom, width, height))
        return res

    def _context_to_idx(self, ctx_docnames: t_ctx)->np.array:
        if len(ctx_docnames) == 0:
            idx = np.ones(self.nb_embeddings, dtype=np.bool)
        else:
            rev_idx = self.get_docname_reverse_index()  # todo(anguelos) cache reverse index
            idx = np.zeros(self.nb_embeddings, dtype=np.bool)
            for ctx_docname in ctx_docnames:
                idx = idx | (self.doccodes == rev_idx[ctx_docname])
        return idx

    def _retrieve_euclidean(self, query_embedding: np.array, ctx_docnames: t_ctx)->Tuple[np.array, np.array]:
        ctx_idx = self._context_to_idx(ctx_docnames)
        print("ctx_idx.shape", ctx_idx.shape)
        print("ctx_idx.sum()", ctx_idx.sum())
        embeddings = self.embeddings[ctx_idx, :]
        reversed_ctx_idx = self.idx[ctx_idx]
        subtracted = embeddings - query_embedding
        similarity = (subtracted * subtracted).sum(axis=1) ** .5
        sorted_ctx_idx = np.argsort(similarity)
        response_idx = reversed_ctx_idx[sorted_ctx_idx]
        return response_idx, similarity

    def _generate_perdoc_occurence_limit(self, response_idx:np.array, max_responces_perdoc:int)->np.array:
        print("max_respoces:",max_responces_perdoc)
        if max_responces_perdoc <=0:
            print("response_idx.shape", response_idx.shape)
            filter = np.ones(response_idx.shape, dtype=np.bool)
            print("filter.shape", filter.shape)
        else:
            response_doccodes = self.doccodes[response_idx]
            occurences = np.zeros([response_idx.shape[0], self.docnames.shape[0]])
            occurences[np.arange(response_idx.shape[0], dtype=int), response_doccodes] = 1
            occurence_count = (occurences.cumsum(axis=0) * occurences).sum(axis=1)
            filter = occurence_count <= max_responces_perdoc
        return filter

    def search(self, query_embedding: np.array, ctx_docnames: t_ctx, max_responces:int, max_occurence_per_document: int)->List:
        if self.metric=="euclidean":
            responce_idx, similarity = self._retrieve_euclidean(query_embedding, ctx_docnames)
        else:
            raise NotImplementedError(self.metric)
        print("responce1:", responce_idx)
        responce_idx, similarity = responce_idx[:max_responces], similarity[:max_responces]
        filter_by_doc_occurences = self._generate_perdoc_occurence_limit(responce_idx, max_occurence_per_document)
        print("filter:",filter_by_doc_occurences)
        responce_idx, similarity = responce_idx[filter_by_doc_occurences], similarity[filter_by_doc_occurences]
        print("responce2:", responce_idx)
        return self._idx_to_response(responce_idx)

    def save(self, path):
        with open(path, "wb") as fd:
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

    def set_random_data(self, max_documents, page_width, page_height, min_word_height, max_word_height, min_word_width, max_word_width, doc_glob, doc_root):
        print("Creating random data ... ", end="")
        t = time.time()
        names = sorted([path[len(doc_root):] for path in glob.glob(doc_glob)])
        names = names[:max_documents]
        self._reset_num_documents(len(names))
        for n in range(self.nb_documents):
            self.docnames[n] = names[n]

        docids_pagenums = []
        name_to_code = self.get_docname_reverse_index()
        for doc in names:
            for path in glob.glob(doc_root+doc+"/*.jp2"):
                doc_id = os.path.dirname(path)[len(doc_root):]
                filename = os.path.basename(path)
                page_num = filename.split(".")[0].split("!")[0].split("_")[-1]
                page_num = int(page_num)
                docids_pagenums.append((name_to_code[doc_id], page_num))
        docids_pagenums = np.array(docids_pagenums)

        self.embeddings[:, :self.embedding_size//2] = np.random.rand(self.nb_embeddings, self.embedding_size//2)
        self.embeddings[:, self.embedding_size // 2:] = np.random.rand(self.nb_embeddings, self.embedding_size - self.embedding_size // 2)
        self.left = np.random.randint(0, page_width, self.nb_embeddings)
        self.top = np.random.randint(0, page_height, self.nb_embeddings)
        self.right = self.left + np.random.randint(min_word_width, max_word_width,
                                                                          self.nb_embeddings)
        self.bottom = self.top + np.random.randint(min_word_height,
                                                                          max_word_height, self.nb_embeddings)
        self.image_widths[:] = page_width
        self.image_heights[:] = page_height

        pages = np.random.randint(0, docids_pagenums.shape[0], self.nb_embeddings)
        print("name_to_code", name_to_code)
        print("pages", np.unique(pages))
        print("docids_pagenums:", np.unique(docids_pagenums[:, 0]))
        self.doccodes = docids_pagenums[pages, 0]
        print("doccodes", self.doccodes)
        self.pagecodes = docids_pagenums[pages, 1]
        print("pagecodes", self.pagecodes)
        print("Done!")