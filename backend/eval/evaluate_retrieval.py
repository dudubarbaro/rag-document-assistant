"""Avalia a qualidade do RETRIEVAL (a parte que mais influencia um RAG).

Para cada pergunta do dataset, verifica se algum dos top-k chunks contém a
resposta esperada. Métricas:
- hit rate@k: % de perguntas em que a resposta apareceu nos k primeiros chunks
- MRR (Mean Reciprocal Rank): média de 1/posição do primeiro chunk correto

Uso (na pasta backend):
    python -m eval.evaluate_retrieval
    LLM_PROVIDER=openai python -m eval.evaluate_retrieval --chunk-size 500 --k 3
"""

import argparse
import json
from pathlib import Path

from app.config import Settings
from app.ingestion import ingest_file
from app.providers import get_embeddings
from app.store import InMemoryRegistry, LocalVectorStore

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    settings = Settings()
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else settings.chunk_overlap

    # store em memória isolado: a avaliação não mexe na base real
    store = LocalVectorStore(get_embeddings(settings))
    for path in sorted((ROOT / "samples").glob("*.md")):
        ingest_file(
            path.name, path.read_bytes(), store, InMemoryRegistry(), chunk_size, chunk_overlap
        )

    dataset = json.loads((Path(__file__).parent / "dataset.json").read_text(encoding="utf-8"))
    hits, reciprocal_ranks = 0, []
    for item in dataset:
        docs = store.similarity_search(item["question"], k=args.k)
        rank = next(
            (i for i, d in enumerate(docs, start=1) if item["expected"] in d.page_content), None
        )
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        print(f"{'OK ' if rank else 'MISS'} rank={rank or '-'}  {item['question']}")

    n = len(dataset)
    print(
        f"\nprovedor={settings.llm_provider} chunk_size={chunk_size} k={args.k}\n"
        f"hit rate@{args.k} = {hits / n:.0%}   MRR = {sum(reciprocal_ranks) / n:.2f}"
    )


if __name__ == "__main__":
    main()
