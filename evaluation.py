import Levenshtein

def calculate_wer(reference, predicted):
    """
    Calculate Word Error Rate (WER) between reference and predicted sentences.

    Args:
        reference (str or list): Reference sentence(s).
        predicted (str or list): Predicted sentence(s).

    Returns:
        float: WER score.

    """
    
    _reference = reference if isinstance(reference, list) else [reference]
    _predicted = predicted if isinstance(predicted, list) else [predicted]

    def _calc_wer(reference, predicted):
        reference_tokens = reference.split()
        predicted_tokens = predicted.split()

        # Calculate Levenshtein distance
        distance = Levenshtein.distance(reference_tokens, predicted_tokens)

        # Calculate WER
        wer = distance / len(reference_tokens)
        return wer
    
    return sum([_calc_wer(ref, pred) for ref, pred in zip(_reference, _predicted)]) / len(_reference)


if __name__ == "__main__":
    reference_transcript = "The quick brown fox jumped over the lazy dog."

    print_wer = lambda wer: print(f"Word Error Rate (WER): {wer:.4f}")

    print_wer(calculate_wer(reference_transcript, "The quick brown fox jumped over the lazy dog.")) #original
    print_wer(calculate_wer(reference_transcript, "The brown fox jumped over the lazy dog."))       #deletion
    print_wer(calculate_wer(reference_transcript, "The brown quick fox jumped over the lazy dog.")) #replacement
    print_wer(calculate_wer(reference_transcript, "The brone quick fox jumped over the lazy dog.")) #replacement + mistake
    print_wer(calculate_wer(reference_transcript, "The quick broan fox jumps over the lacy dog."))  #mistakes