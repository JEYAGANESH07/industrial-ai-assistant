import re

from wordfreq import zipf_frequency


def word_score(word):
    """
    Return how likely a string is to be
    a normal English word.
    """

    return zipf_frequency(
        word.lower(),
        "en"
    )


def should_join(parts):
    """
    Decide whether multiple adjacent pieces
    are likely to be one incorrectly split word.
    """

    if len(parts) < 2:
        return False

    joined = "".join(parts).lower()

    # Ignore extremely long candidates
    if len(joined) > 30:
        return False

    joined_score = word_score(
        joined
    )

    # Joined word must have reasonable
    # English-word frequency.
    if joined_score < 3.0:
        return False

    part_scores = [
        word_score(part)
        for part in parts
    ]

    # If one of the fragments is very unlikely
    # to be a real word, this is strong evidence
    # that the PDF split a word incorrectly.
    has_weak_fragment = any(
        score < 2.5
        for score in part_scores
    )

    if has_weak_fragment:
        return True

    # Otherwise require the combined word to be
    # substantially more likely than its pieces.
    highest_part_score = max(
        part_scores
    )

    if (
        joined_score
        >= highest_part_score + 1.0
    ):
        return True

    return False


def repair_split_words(text):
    """
    Repair arbitrary PDF split-word artifacts.

    Supports:

        ass embly
        relati on
        opera tions
        trans form ed

    without maintaining a hard-coded list.
    """

    words = text.split()

    result = []

    i = 0

    while i < len(words):

        best_match = None
        best_length = 0
        best_score = 0

        # Try joining 2, 3 and 4 adjacent words.
        for number_of_parts in range(
            2,
            5
        ):

            if (
                i + number_of_parts
                > len(words)
            ):
                break

            candidate_parts = words[
                i:i + number_of_parts
            ]

            # Only work with simple alphabetic
            # fragments.
            if not all(
                part.isalpha()
                for part in candidate_parts
            ):
                continue

            if not should_join(
                candidate_parts
            ):
                continue

            joined = "".join(
                candidate_parts
            )

            score = word_score(
                joined
            )

            if score > best_score:

                best_match = joined
                best_length = (
                    number_of_parts
                )
                best_score = score

        # If a good multi-part word was found
        if best_match is not None:

            result.append(
                best_match
            )

            i += best_length

        else:

            result.append(
                words[i]
            )

            i += 1

    return " ".join(
        result
    )


def clean_pdf_text(text):
    """
    General PDF text cleaning.
    """

    # Normalize line endings
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Repair words broken across lines
    #
    # Example:
    #
    # manufac-
    # turing
    #
    # becomes:
    #
    # manufacturing
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # Convert remaining newlines
    # to spaces.
    text = re.sub(
        r"\s*\n\s*",
        " ",
        text
    )

    # Remove repeated spaces.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.strip()

    # Repair arbitrary split words.
    text = repair_split_words(
        text
    )

    return text


# ------------------------------------------------
# TEST CASES
# ------------------------------------------------

test_cases = [
    "The ass embly operation is complete.",
    "There is a relati on between the systems.",
    "The opera tions were completed.",
    "The compo nent was replaced.",
    "The v olumes were measured.",
    "The Wo rking principle is explained.",
    "The machine tool is used.",
    "The heat treating process is important.",
    "The rapid prototyping process is useful.",
    "The power plant operates continuously.",
    "The data base contains information.",
    "The trans form ed material was inspected.",
]


print()
print(
    "=============================="
)

print(
    "SPLIT-WORD CLEANER TEST"
)

print(
    "=============================="
)

print()


for original in test_cases:

    cleaned = clean_pdf_text(
        original
    )

    print(
        "BEFORE:"
    )

    print(
        original
    )

    print(
        "AFTER:"
    )

    print(
        cleaned
    )

    print(
        "------------------------------"
    )