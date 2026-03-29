import ir_datasets
import statistics
import collections

def word_count(text: str) -> int:
    return len(text.split()) if text else 0

def char_count(text: str) -> int:
    return len(text) if text else 0

def describe(values: list, label: str) -> dict:
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)
    mean = sum(values_sorted) / n
    median = values_sorted[n // 2]
    stdev = statistics.stdev(values_sorted) if n > 1 else 0
    return {
        "label": label,
        "n": n,
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "mean": round(mean, 2),
        "median": median,
        "stdev": round(stdev, 2),
        "p25": values_sorted[int(n * 0.25)],
        "p75": values_sorted[int(n * 0.75)],
        "p95": values_sorted[int(n * 0.95)],
    }

def print_stats(d: dict):
    print(f"  n={d['n']:,}  min={d['min']}  max={d['max']}  "
          f"mean={d['mean']}  median={d['median']}  stdev={d['stdev']}")
    print(f"  p25={d['p25']}  p75={d['p75']}  p95={d['p95']}")




corpus_ds = ir_datasets.load("trec-tot/2023")
train_ds  = ir_datasets.load("trec-tot/2023/train")
dev_ds    = ir_datasets.load("trec-tot/2023/dev")

print("Loaded: trec-tot/2023 | trec-tot/2023/train | trec-tot/2023/dev")

print("Key statistics: ")

doc_word_counts      = []
doc_char_counts      = []
section_counts       = []
infobox_counts       = []
wikidata_class_counts = []

missing_text   = 0
missing_title  = 0
duplicate_ids  = collections.Counter()

doc_ids = set()

for doc in corpus_ds.docs_iter():
    duplicate_ids[doc.doc_id] += 1
    doc_ids.add(doc.doc_id)

    # Text length
    text = doc.text or ""
    if not text.strip():
        missing_text += 1
    doc_word_counts.append(word_count(text))
    doc_char_counts.append(char_count(text))

    # Title
    if not doc.page_title or not doc.page_title.strip():
        missing_title += 1

    # Sections
    n_sections = len(doc.sections) if doc.sections else 0
    section_counts.append(n_sections)

    # Infoboxes
    n_infoboxes = len(doc.infoboxes) if doc.infoboxes else 0
    infobox_counts.append(n_infoboxes)

    # Wikidata classes
    n_classes = len(doc.wikidata_classes) if doc.wikidata_classes else 0
    wikidata_class_counts.append(n_classes)

n_docs = len(doc_word_counts)
n_duplicates = sum(1 for c in duplicate_ids.values() if c > 1)

print(f"\nTotal documents : {n_docs:,}")
print(f"Duplicate doc IDs: {n_duplicates}")
print(f"Missing text    : {missing_text}")
print(f"Missing title   : {missing_title}")

print("\n-- Document text length (words) --")
print_stats(describe(doc_word_counts, "doc_words"))

print("\n-- Document text length (chars) --")
print_stats(describe(doc_char_counts, "doc_chars"))

print("\n-- Number of sections per doc --")
print_stats(describe(section_counts, "sections"))

print("\n-- Number of infoboxes per doc --")
print_stats(describe(infobox_counts, "infoboxes"))

print("\n-- Number of Wikidata classes per doc --")
print_stats(describe(wikidata_class_counts, "wikidata_classes"))

# Section length breakdown
zero_sections = sum(1 for s in section_counts if s == 0)
print(f"\nDocs with 0 sections : {zero_sections:,} ({100*zero_sections/n_docs:.1f}%)")
print(f"Docs with 0 infoboxes: {sum(1 for x in infobox_counts if x == 0):,} ({100*sum(1 for x in infobox_counts if x==0)/n_docs:.1f}%)")


print("Query Analysis: ")
for split_name, split_ds in [("TRAIN", train_ds), ("DEV", dev_ds)]:
    q_word_counts  = []
    q_char_counts  = []
    q_domains      = collections.Counter()
    q_annot_counts = []
    missing_q_text = 0
    q_ids          = set()

    for q in split_ds.queries_iter():
        q_ids.add(q.query_id)
        text = q.text or ""
        if not text.strip():
            missing_q_text += 1
        q_word_counts.append(word_count(text))
        q_char_counts.append(char_count(text))
        q_domains[q.domain or "UNKNOWN"] += 1
        n_annot = len(q.sentence_annotations) if q.sentence_annotations else 0
        q_annot_counts.append(n_annot)

    print(f"\n[{split_name}]")
    print(f"  Total queries     : {len(q_ids):,}")
    print(f"  Missing text      : {missing_q_text}")

    print("\n  -- Query length (words) --")
    print_stats(describe(q_word_counts, "query_words"))

    print("\n  -- Query length (chars) --")
    print_stats(describe(q_char_counts, "query_chars"))

    print("\n  -- Sentence annotations per query --")
    print_stats(describe(q_annot_counts, "annotations"))

    print(f"\n  -- Domain distribution (top 10) --")
    for domain, count in q_domains.most_common(10):
        pct = 100 * count / len(q_ids)
        print(f"    {domain:<30} {count:>5}  ({pct:.1f}%)")




print("Summary: ")
checks = [
    ("Duplicate document IDs in corpus",      n_duplicates),
    ("Documents with empty text",             missing_text),
    ("Documents with empty title",            missing_title),
    ("Documents with 0 sections",             zero_sections),
]

for label, value in checks:
    print(f"  {label:<45} {value:>6}")


