import re

multitap_encode = {
    'A': '2^1', 'B': '2^2', 'C': '2^3',
    'D': '3^1', 'E': '3^2', 'F': '3^3',
    'G': '4^1', 'H': '4^2', 'I': '4^3',
    'J': '5^1', 'K': '5^2', 'L': '5^3',
    'M': '6^1', 'N': '6^2', 'O': '6^3',
    'P': '7^1', 'Q': '7^2', 'R': '7^3', 'S': '7^4',
    'T': '8^1', 'U': '8^2', 'V': '8^3',
    'W': '9^1', 'X': '9^2', 'Y': '9^3', 'Z': '9^4',
}

multitap_decode = {v: k for k, v in multitap_encode.items()}

def encode_multitap(text):
    text = text.upper()
    encoded_text = ''
    for char in text:
        if char in multitap_encode:
            encoded_text += multitap_encode[char]
    return encoded_text

def decode_multitap(encoded_text):
    decoded_text = ''
    matches = re.findall(r'\d\^\d|\d', encoded_text)
    for match in matches:
        if match in multitap_decode:
            decoded_text += multitap_decode[match]
    return decoded_text
