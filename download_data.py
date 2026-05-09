"""
Dataset download and preparation script for TREC 2023 ToT retrieval project.

Downloads and converts to JSONL format expected by dataset.py:
  data/
    trec/
      train_queries.jsonl
      dev_queries.jsonl
      docs.jsonl
      train_qrels.txt
      dev_qrels.txt
    reddit/
      queries.jsonl

Usage:
  pip install ir_datasets tqdm requests
  python download_data.py --output-dir data
  python download_data.py --output-dir data --skip-docs   # skip 231K doc collection (slow)
"""

import os
import json
import argparse
import requests
import zipfile
import io
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'  Saved {len(records)} records → {path}')


def download_file(url, dest_path, desc='Downloading'):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        desc=desc, total=total, unit='B', unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f'  Downloaded → {dest_path}')


# ---------------------------------------------------------------------------
# TREC 2023 ToT via ir_datasets
# ---------------------------------------------------------------------------

def download_trec_tot(output_dir, skip_docs=False):
    """
    Downloads TREC 2023 ToT dataset using the ir_datasets library.
    ir_datasets handles authentication and caching automatically.

    Install: pip install ir_datasets
    Dataset ID: trec-tot/2023/train  and  trec-tot/2023/dev
    """
    try:
        import ir_datasets
    except ImportError:
        print('[ERROR] ir_datasets not installed. Run: pip install ir_datasets')
        print_manual_trec_instructions()
        return False

    trec_dir = os.path.join(output_dir, 'trec')
    os.makedirs(trec_dir, exist_ok=True)

    # --- queries and qrels ---
    for split in ['train', 'dev']:
        dataset_id = f'trec-tot/2023/{split}'
        print(f'\nLoading {dataset_id} ...')

        try:
            dataset = ir_datasets.load(dataset_id)
        except Exception as e:
            print(f'[ERROR] Could not load {dataset_id}: {e}')
            print_manual_trec_instructions()
            return False

        # queries — extract sentence annotations if available
        queries = []
        for q in tqdm(dataset.queries_iter(), desc=f'  Queries ({split})'):
            sentences = []
            if hasattr(q, 'sentences') and q.sentences:
                sentences = list(q.sentences)
            elif hasattr(q, 'text') and q.text:
                # fall back: split on period
                sentences = [s.strip() for s in q.text.split('.') if s.strip()]

            queries.append({
                'id':        q.query_id,
                'text':      q.text if hasattr(q, 'text') else ' '.join(sentences),
                'sentences': sentences,
            })

        save_jsonl(queries, os.path.join(trec_dir, f'{split}_queries.jsonl'))

        # qrels
        qrel_path = os.path.join(trec_dir, f'{split}_qrels.txt')
        with open(qrel_path, 'w') as f:
            for qrel in tqdm(dataset.qrels_iter(), desc=f'  Qrels ({split})'):
                f.write(f'{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n')
        print(f'  Saved qrels → {qrel_path}')

    # --- document collection ---
    if not skip_docs:
        print('\nLoading document collection (231K Wikipedia docs — this may take a while)...')
        try:
            doc_dataset = ir_datasets.load('trec-tot/2023/train')  # docs shared across splits
            docs = []
            for doc in tqdm(doc_dataset.docs_iter(), desc='  Documents'):
                docs.append({
                    'id':       doc.doc_id,
                    'title':    doc.title    if hasattr(doc, 'title')    else '',
                    'text':     doc.text     if hasattr(doc, 'text')     else '',
                    'sections': doc.sections if hasattr(doc, 'sections') else [],
                })
            save_jsonl(docs, os.path.join(trec_dir, 'docs.jsonl'))
        except Exception as e:
            print(f'[ERROR] Could not load documents: {e}')
            print_manual_trec_instructions()
            return False
    else:
        print('\n  Skipping document collection (--skip-docs flag set).')

    print('\nTREC 2023 ToT dataset ready.')
    return True


def print_manual_trec_instructions():
    print("""
  Manual download instructions for TREC 2023 ToT:
  ─────────────────────────────────────────────────
  1. Register at https://trec.nist.gov and request access to the 2023 ToT track data.
  2. Download the following files from the TREC data portal:
       - topics (queries with sentence annotations)
       - qrels
       - document collection (Wikipedia dump)
  3. Convert them to the JSONL format below and place in data/trec/:

     train_queries.jsonl  → [{id, text, sentences: [...]}, ...]
     dev_queries.jsonl    → [{id, text, sentences: [...]}, ...]
     docs.jsonl           → [{id, title, text}, ...]
     train_qrels.txt      → standard TREC format: qid 0 docid rel
     dev_qrels.txt        → standard TREC format: qid 0 docid rel
    """)


# ---------------------------------------------------------------------------
# Reddit ToT dataset
# ---------------------------------------------------------------------------

def _extract_domain(categories):
    """Map Reddit category tags to a simple domain string."""
    if not categories:
        return 'unknown'
    joined = ' '.join(categories).lower()
    if 'movie' in joined or 'film' in joined or 'anime' in joined:
        return 'movie'
    if 'book' in joined or 'novel' in joined or 'comic' in joined:
        return 'book'
    if 'music' in joined or 'song' in joined or 'album' in joined:
        return 'music'
    if 'game' in joined or 'video game' in joined:
        return 'game'
    if 'tv' in joined or 'television' in joined or 'show' in joined:
        return 'tv'
    return categories[0].lower() if categories else 'unknown'


def download_reddit_tot(output_dir):
    """
    Downloads the TOMT-KIS Reddit dataset from HuggingFace:
      webis/tip-of-my-tongue-known-item-search-triplets

    Matches answer documents against the TREC docs collection by doc ID,
    then falls back to Wikipedia title matching if needed.

    Install: pip install datasets
    """
    reddit_dir = os.path.join(output_dir, 'reddit')
    os.makedirs(reddit_dir, exist_ok=True)

    print('\nDownloading Reddit ToT dataset (webis/tip-of-my-tongue-known-item-search-triplets)...')

    try:
        from datasets import load_dataset
    except ImportError:
        print('[ERROR] datasets not installed. Run: pip install datasets')
        print_manual_reddit_instructions()
        return False

    # --- load TREC doc index for answer matching ---
    doc_path = os.path.join(output_dir, 'trec', 'docs.jsonl')
    print('  Building doc index from TREC collection for answer matching...')
    id_set     = set()   # string doc IDs
    title_to_id = {}     # Wikipedia title → doc_id (fallback)
    with open(doc_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            id_set.add(str(d['id']))
            if d.get('title'):
                title_to_id[d['title'].lower()] = str(d['id'])
    print(f'  Doc index ready: {len(id_set):,} docs, {len(title_to_id):,} titles')

    def resolve_doc_id(docno, wiki_url):
        """Return matching TREC doc_id or None."""
        # primary: docno_positive as string
        if str(docno) in id_set:
            return str(docno)
        # fallback: extract title from Wikipedia URL
        if wiki_url:
            title = wiki_url.rstrip('/').split('/')[-1].replace('_', ' ')
            match = title_to_id.get(title.lower())
            if match:
                return match
        return None

    try:
        print('  Loading HuggingFace dataset...')
        hf_dataset = load_dataset('webis/tip-of-my-tongue-known-item-search-triplets')
    except Exception as e:
        print(f'  HuggingFace load failed: {e}')
        print_manual_reddit_instructions()
        return False

    records = []
    skipped = 0

    for split_name, split_data in hf_dataset.items():
        for ex in tqdm(split_data, desc=f'  {split_name}'):
            doc_id = resolve_doc_id(
                ex.get('docno_positive', ''),
                ex.get('url_wikipedia_positive', '')
            )
            if doc_id is None:
                skipped += 1
                continue

            text      = ex.get('query', '')
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            domain    = _extract_domain(ex.get('categories', []))

            records.append({
                'id':        str(ex.get('qid', '')),
                'text':      text,
                'sentences': sentences,
                'answer_id': doc_id,
                'domain':    domain,
            })

    if skipped:
        print(f'  Skipped {skipped} queries whose answer doc is not in the TREC collection.')

    if not records:
        print('  No matching records found. The TREC doc IDs may not overlap with this dataset.')
        print_manual_reddit_instructions()
        return False

    save_jsonl(records, os.path.join(reddit_dir, 'queries.jsonl'))
    print(f'Reddit ToT dataset ready: {len(records)} queries saved.')
    return True


def print_manual_reddit_instructions():
    print("""
  Manual download instructions for Reddit ToT dataset:
  ─────────────────────────────────────────────────────
  1. The TOMT-KIS dataset is derived from Reddit r/tipofmytongue.
  2. Check the TREC 2023 ToT track GitHub or the paper's supplementary:
       'Generalizable Tip-of-the-Tongue Retrieval with LLM Re-ranking'
  3. Convert to JSONL format and place in data/reddit/:

     queries.jsonl → [{id, text, sentences: [...], answer_id, domain}, ...]

     The answer_id must match a document id in data/trec/docs.jsonl.
    """)


# ---------------------------------------------------------------------------
# Verify data structure
# ---------------------------------------------------------------------------

def verify(output_dir):
    print('\nVerifying downloaded files ...')

    expected = {
        'data/trec/train_queries.jsonl': 'TREC train queries',
        'data/trec/dev_queries.jsonl':   'TREC dev queries',
        'data/trec/docs.jsonl':          'Document collection',
        'data/trec/train_qrels.txt':     'TREC train qrels',
        'data/trec/dev_qrels.txt':       'TREC dev qrels',
    }

    all_ok = True
    for rel_path, name in expected.items():
        full_path = os.path.join(output_dir, rel_path.replace('data/', ''))
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            # count lines for jsonl
            with open(full_path) as f:
                n = sum(1 for _ in f)
            print(f'  [OK] {name:<30} {n:>8} lines  ({size / 1e6:.1f} MB)')
        else:
            print(f'  [MISSING] {name}  →  {full_path}')
            all_ok = False

    reddit_path = os.path.join(output_dir, 'reddit', 'queries.jsonl')
    if os.path.exists(reddit_path):
        with open(reddit_path) as f:
            n = sum(1 for _ in f)
        print(f'  [OK] {"Reddit queries":<30} {n:>8} lines')
    else:
        print(f'  [MISSING] Reddit queries  →  {reddit_path}')

    if all_ok:
        print('\n  All required files present. Ready to train.\n')
    else:
        print('\n  Some files are missing. See instructions above.\n')


# ---------------------------------------------------------------------------
# Argument parsing and entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Download TREC 2023 ToT and Reddit ToT datasets')
    p.add_argument('--output-dir',   default='data',  help='Root output directory')
    p.add_argument('--skip-docs',    action='store_true',
                   help='Skip downloading the 231K document collection (large file)')
    p.add_argument('--skip-reddit',  action='store_true',
                   help='Skip downloading the Reddit ToT dataset')
    p.add_argument('--skip-trec',    action='store_true',
                   help='Skip downloading TREC queries/qrels (e.g. already downloaded)')
    p.add_argument('--reddit-only',  action='store_true',
                   help='Download only the Reddit dataset (implies --skip-trec --skip-docs)')
    p.add_argument('--verify-only',  action='store_true',
                   help='Only check if all files are present, do not download')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.verify_only:
        verify(args.output_dir)
    else:
        print('TREC 2023 Tip-of-the-Tongue Dataset Downloader')
        print('=' * 50)

        skip_trec = args.skip_trec or args.reddit_only
        skip_docs = args.skip_docs or args.reddit_only

        # TREC dataset
        if not skip_trec:
            download_trec_tot(args.output_dir, skip_docs=skip_docs)

        # Reddit dataset
        if not args.skip_reddit:
            download_reddit_tot(args.output_dir)

        # verify everything is in place
        verify(args.output_dir)
