import random
from typing import Callable, List
import functools

LETTERS = ['J', 'D', 'W', 'O', 'T', 'R', 'A', 'C', 'X', 'Q', 'M', 'F', 'Y', 
            'E', 'Z', 'G', 'U', 'K', 'P', 'V', 'B', 'S', 'H', 'N', 'L', 'I']

LETTER_TO_NUM_MAP = {}
for i, letter in enumerate(LETTERS):
    LETTER_TO_NUM_MAP[letter] = i

input_key = [9, 25, 44, 38, 40, 22, 11, 36, 13, 39, 18, 42, 10, 53, 26, 12, 1, 16, 3, 43, 37, 17, 30, 4, 28, 48, 27, 41, 32, 15, 47, 29, 20, 51, 6, 7, 52, 34, 35, 5, 50, 9, 54, 46, 23, 31, 24, 14, 8, 33, 2, 49, 45, 21]

def solitaire_encrypt(msg: str) -> str:
    return encrypt(input_key, msg)

def solitaire_decrypt(msg: str) -> str:
    return decrypt(input_key, msg)

def encrypt(key: list, msg: str) -> str:
    return transform(key, msg, lambda n, k: n + k)

def decrypt(key: list, msg: str) -> str:
    return transform(key, msg, lambda n, k: n - k)

def transform(cards: List[int], msg: str, combine: Callable) -> str:
    input = format_input(msg)
    cards = list(cards)
    run_keystream_sequence = compose(count_cut,
                                triple_cut_by_jokers,
                                move_joker_b,
                                move_joker_a)
    output = []
    while len(output) < len(input):
        cards = run_keystream_sequence(cards)
        ks_val = get_keystream_value(cards)
        if is_joker(ks_val, cards):
            continue
        current_letter = input[len(output)]
        value = combine(LETTER_TO_NUM_MAP[current_letter], ks_val)
        output.append(number_to_letter(value))

    return ''.join(output)

def generate_cards(suits=4) -> List[int]:
    jokers = 2
    deck_size = suits * 13 + jokers
    cards = list(range(1, deck_size + 1))
    random.shuffle(cards)
    return cards

def triple_cut_by_jokers(cards: List[int]) -> list:
    joker_a = len(cards) - 1
    joker_b = len(cards)
    return triple_cut((cards.index(joker_a), cards.index(joker_b)), cards)

def move_joker_a(cards: List[int]) -> list:
    return move_joker(len(cards) - 1, cards)

def move_joker_b(cards: List[int]) -> list:
    return move_joker(len(cards), cards)

def get_keystream_value(cards: List[int]) -> int:
    index = cards[0] if not is_joker(cards[0], cards) else len(cards) - 1
    return cards[index]

def is_joker(value: int, cards: List[int]) -> bool:
    return value > len(cards) - 2

def move_joker(joker: int, cards: List[int]) -> List[int]:
    def wraparound(n: int) -> int:
        if n >= len(cards):
            return n % len(cards) + 1
        return n

    cards = list(cards)
    jump = 2 if joker is len(cards) else 1
    index = cards.index(joker)
    cards.insert(wraparound(index + jump), cards.pop(index))
    return cards

def triple_cut(cut_indices: tuple, cards: list) -> List[int]:
    lower, higher = sorted(cut_indices)
    return cards[higher + 1:] + cards[lower:higher + 1] + cards[:lower]

def count_cut(cards: List[int]) -> List[int]:
    last = len(cards) - 1
    value = cards[last]
    if is_joker(value, cards):
        return list(cards)
    return cards[value:last] + cards[:value] + [cards[last]]

def number_to_letter(n: int) -> str:
    return LETTERS[n % len(LETTERS)]

def format_input(msg: str) -> List[str]:
    return [char for char in msg.upper() if char in LETTER_TO_NUM_MAP]

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions)