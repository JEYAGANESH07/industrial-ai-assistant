import re
from pathlib import Path
from pypdf import PdfReader


# Patterns that may indicate PDF text-extraction problems.
SUSPICIOUS_PATTERNS = {
    "Unexpected space after capital letter": re.compile(
        r"\b[A-Z]\s+[a-z]{2,}\b"
    ),
    "Possible split inside a word": re.compile(
        r"\b[a-z]{3,}\s+[a-z]{2,}\b"
    ),
    "Unusual encoded-looking sequence": re.compile(
        r"(?:/[A-Z0-9]{2,}){3,}"
    ),
}


def inspect_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))

    print(f"\nPDF: {pdf_path.name}")
    print(f"Pages: {len(reader.pages)}")

    total_findings = 0

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_findings = []

        # Encoded-looking sequences are particularly worth reviewing.
        for match in SUSPICIOUS_PATTERNS[
            "Unusual encoded-looking sequence"
        ].finditer(text):
            page_findings.append(
                ("Unusual encoded-looking sequence", match)
            )

        # Capital-letter splits, such as "T otal" or "W elding".
        for match in SUSPICIOUS_PATTERNS[
            "Unexpected space after capital letter"
        ].finditer(text):
            page_findings.append(
                ("Unexpected space after capital letter", match)
            )

        if page_findings:
            print(f"\n--- Page {page_number} ---")

            for issue, match in page_findings:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].replace("\n", " ")

                print(f"Issue:   {issue}")
                print(f"Text:    {match.group()}")
                print(f"Context: {context}\n")

            total_findings += len(page_findings)

    print(f"Finished. Candidates for review: {total_findings}")
    print("These are candidates, not confirmed errors.")


if __name__ == "__main__":
    filename = input(
        "Enter the full path to the PDF: "
    ).strip().strip('"')

    pdf_path = Path(filename)

    if not pdf_path.is_file():
        print(f"File not found: {pdf_path}")
    else:
        inspect_pdf(pdf_path)