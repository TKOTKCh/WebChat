import torch
from transformers import AutoTokenizer, AutoModel
import os
import re

from typing import Optional, Union, List, Dict, Tuple, Iterable, Callable, Any

from mymodel.retriever.filtering.contriever import Contriever


class ContrieverScorer:
    def __init__(self, retriever_ckpt_path, device=None, max_batch_size=400) -> None:

        self.contriever=Contriever.from_pretrained("mymodel/facebook/mcontriever-msmarco")
        self.tokenizer = AutoTokenizer.from_pretrained("mymodel/facebook/mcontriever-msmarco")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if not device else device
        self.contriever=self.contriever.to(self.device).eval()
        assert max_batch_size > 0
        self.max_batch_size = max_batch_size
    
    def get_embeddings(self, sentences: List[str]) -> torch.Tensor:
        # Tokenization and Inference
        torch.cuda.empty_cache()
        with torch.no_grad():
            inputs = self.tokenizer(sentences, padding=True,
                                    truncation=True, return_tensors='pt')
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)

            sentence_embeddings=self.contriever(**inputs)
            return sentence_embeddings

    def score_documents_on_query(self, query: str, documents: List[str]) -> torch.Tensor:
        query_embedding = self.get_embeddings([query])[0]
        document_embeddings = self.get_embeddings(documents)
        return query_embedding@document_embeddings.t()

    def select_topk(self, query: str, documents: List[str], k=1):
        """
        Returns:
            `ret`: `torch.return_types.topk`, use `ret.values` or `ret.indices` to get value or index tensor
        """
        scores = []
        for i in range((len(documents) + self.max_batch_size - 1) // self.max_batch_size):
            scores.append(self.score_documents_on_query(query, documents[self.max_batch_size*i:self.max_batch_size*(i+1)]).to('cpu'))
        scores = torch.concat(scores)
        return scores.topk(min(k, len(scores)))


class ReferenceFilter:
    def __init__(self, retriever_ckpt_path, device=None, max_batch_size=400) -> None:
        self.scorer = ContrieverScorer(retriever_ckpt_path, device, max_batch_size)

    def produce_references(self, query, paragraphs: List[Dict[str, str]], topk=5) -> List[Dict[str, str]]:
        """Individually calculate scores of each sentence, and return `topk`. paragraphs should be like a list of {title, url, text}."""
        # paragraphs = self._pre_filter(paragraphs)
        texts = [item['text'] for item in paragraphs]
        topk = self.scorer.select_topk(query, texts, topk)
        indices = list(topk.indices.detach().cpu().numpy())
        return [paragraphs[idx] for idx in indices]


