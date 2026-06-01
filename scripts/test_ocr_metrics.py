from document_checker.ocr.metrics import (
    normalize_text,
    levenshtein_distance,
    character_accuracy,
    word_accuracy,
    word_error_rate,
)


def main() -> None:
    reference = "This is a simple test sentence."
    hypothesis_good = "This is a simple test sentence."
    hypothesis_noisy = "Th1s is simple tst sentnce"

    print("Reference:        ", reference)
    print("Hypothesis (good):", hypothesis_good)
    print("Hypothesis (noisy):", hypothesis_noisy)
    print()

    print("Normalized reference:       ", normalize_text(reference))
    print("Normalized hypothesis good: ", normalize_text(hypothesis_good))
    print("Normalized hypothesis noisy:", normalize_text(hypothesis_noisy))
    print()

    print("Levenshtein (ref vs good): ", levenshtein_distance(reference, hypothesis_good))
    print("Levenshtein (ref vs noisy):", levenshtein_distance(reference, hypothesis_noisy))
    print()

    print("Character accuracy (good): ", character_accuracy(reference, hypothesis_good))
    print("Character accuracy (noisy):", character_accuracy(reference, hypothesis_noisy))
    print()

    print("Word accuracy (good): ", word_accuracy(reference, hypothesis_good))
    print("Word accuracy (noisy):", word_accuracy(reference, hypothesis_noisy))
    print()

    print("WER (good): ", word_error_rate(reference, hypothesis_good))
    print("WER (noisy):", word_error_rate(reference, hypothesis_noisy))


if __name__ == "__main__":
    main()