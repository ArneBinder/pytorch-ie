from pytorch_ie.metrics.statistics import TokenCountCollector


def test_statistics_with_tokenize(document_dataset):
    statistic = TokenCountCollector(
        text_field="text",
        tokenizer="bert-base-uncased",
        tokenizer_kwargs=dict(add_special_tokens=False),
    )
    values = statistic(document_dataset)
    assert values == {
        "test": {"max": 13, "mean": 8.5, "min": 4, "std": 4.5},
        "train": {"max": 14, "mean": 7.75, "min": 4, "std": 3.6314597615834874},
        "val": {"max": 13, "mean": 8.5, "min": 4, "std": 4.5},
    }
