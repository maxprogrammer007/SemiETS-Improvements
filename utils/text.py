def text_to_indices(text, vocab):
    """
    Convert string text into list of character indices
    """
    text = text.lower()
    indices = []

    for ch in text:
        if ch in vocab:
            indices.append(vocab.index(ch))

    return indices
