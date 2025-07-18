# normalize_my_dict.py
"""
A one-time utility to normalize an existing G2P dictionary file that
has syllable/stress markers but lacks space segmentation.

This script reads a .tsv file and ensures that every phoneme string
is correctly segmented with spaces, making it ready for training with the
current G2P pipeline.

Example Input Line:
abá     a.ˈba

Example Output Line:
abá     a . ˈb a

Usage:
    python normalize_my_dict.py --input ./path/to/your/pt_br.tsv --output ./dicts/pt-br_normalized.tsv
"""
import argparse
import regex as re
from tqdm import tqdm

# This regex finds either a special marker OR any single grapheme cluster.
# It's the key to correctly segmenting the phoneme string.
# It matches:
# 1. A syllable boundary `.` OR a stress marker `ˈ`
# 2. OR any other single character/grapheme `\X`
SEGMENTER_REGEX = re.compile(r"[.ˈ]|\X")


def main(args):
    """Main function to read, normalize, and write the dictionary."""
    print(f"Reading from input file: {args.input}")

    try:
        with open(args.input, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input}'")
        return

    print(f"Normalizing {len(lines)} entries...")

    with open(args.output, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines, desc="Processing"):
            # The user's file format has multiple spaces, so we split by any whitespace.
            parts = line.strip().split(None, 1)

            if len(parts) != 2:
                print(f"Skipping malformed line: '{line.strip()}'")
                continue

            word, phonemes = parts

            # 1. Use the regex to find all phonemes and markers as a list of tokens.
            # Example: 'a.ˈba' -> ['a', '.', 'ˈ', 'b', 'a']
            token_list = SEGMENTER_REGEX.findall(phonemes)

            # 2. Join the tokens back together with a single space.
            # Example: ['a', '.', 'ˈ', 'b', 'a'] -> 'a . ˈ b a'
            normalized_phonemes = " ".join(token_list)

            # 3. Write the clean line to the new file.
            f_out.write(f"{word.lower()}\t{normalized_phonemes}\n")

    print(f"\nNormalization complete. Clean dictionary saved to: {args.output}")
    print("You can now use this new file for training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cleans and normalizes a G2P .tsv dictionary file.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input .tsv file (unsegmented, with markers)."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save the clean, normalized, and segmented .tsv file."
    )

    args = parser.parse_args()
    main(args)