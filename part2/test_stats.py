from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

def get_stats(split):
    with open(f"data/{split}.nl") as f:
        questions = [line.strip() for line in f]

    with open(f"data/{split}.sql") as f:
        queries = [line.strip() for line in f]

    nl_tokens = [tokenizer.encode(x, add_special_tokens = True) for x in questions]
    sql_tokens = [tokenizer.encode(x, add_special_tokens = True) for x in queries]

    return {
        "Examples": len(questions),
        "Mean NL length": sum(map(len, nl_tokens)) / len(nl_tokens),
        "Mean SQL length": sum(map(len, sql_tokens)) / len(sql_tokens),
        "NL vocabulary": len(set(t for seq in nl_tokens for t in seq)),
        "SQL vocabulary": len(set(t for seq in sql_tokens for t in seq)),
    }

for split in ["train", "dev"]:
    print(f"\n{split.upper()}")
    for key, value in get_stats(split).items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")


for split in ["train", "dev"]:
    with open(f"data/{split}.sql") as f:
        queries = [line.strip() for line in f]

    lengths = [
        len(tokenizer.encode(q, add_special_tokens=True))
        for q in queries
    ]

    print(f"\n{split.upper()} SQL LENGTHS")
    print("Min:", min(lengths))
    print("Mean:", sum(lengths) / len(lengths))
    print("Max:", max(lengths))
    print("95th percentile:", sorted(lengths)[int(0.95 * len(lengths))])

    longest_idx = lengths.index(max(lengths))

    print("\nLongest query:")
    print(queries[longest_idx])
