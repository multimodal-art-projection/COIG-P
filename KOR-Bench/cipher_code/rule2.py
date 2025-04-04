encryption_table = {
    'A': '!', 'B': '@', 'C': '#', 'D': '$',
    'E': '%', 'F': '^', 'G': '&', 'H': '*',
    'I': '(', 'J': ')', 'K': '_', 'L': '+',
    'M': '=', 'N': '~', 'O': '?', 'P': '/',
    'Q': '0', 'R': ':', 'S': ';', 'T': '<',
    'U': '>', 'V': '1', 'W': '2', 'X': '3',
    'Y': '4', 'Z': '5'
}

decryption_table = {v: k for k, v in encryption_table.items()}

def encrypt(text):
    encrypted_text = "".join(encryption_table.get(char, char) for char in text)
    return encrypted_text

def decrypt(encrypted_text):
    decrypted_text = "".join(decryption_table.get(char, char) for char in encrypted_text)
    return decrypted_text
