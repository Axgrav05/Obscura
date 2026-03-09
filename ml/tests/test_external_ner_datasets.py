from ml.external_ner_datasets import _expand_bio, _map_labels


def test_map_labels_preserves_bio_prefix_for_conll() -> None:
    label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    mapping = {
        "PER": "B-PER",
        "ORG": "B-ORG",
        "LOC": "B-LOC",
        "MISC": "B-MISC",
    }

    mapped = _map_labels(label_names, [1, 2, 3, 4, 5, 6, 7, 8], mapping)

    assert mapped == [
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ]


def test_expand_bio_converts_i_prefix() -> None:
    assert _expand_bio("I-corporation", "B-ORG") == "I-ORG"
    assert _expand_bio("B-group", "B-ORG") == "B-ORG"
    assert _expand_bio("O", "O") == "O"