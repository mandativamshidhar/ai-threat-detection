import os
import argparse
from .eval import mrr_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--num_queries', type=int, default=200)
    args = parser.parse_args()

    db_path = os.path.join(os.path.dirname(__file__), 'metadata.db')
    # reuse benchmark's sampler
    from .benchmark import load_sample_queries
    queries, ids = load_sample_queries(db_path, args.num_queries)
    score = mrr_at_k(queries, ids, k=args.k)
    print(f"MRR@{args.k} over {len(queries)} queries: {score:.4f}")


if __name__ == '__main__':
    main()
