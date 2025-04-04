keyword="RFDDJUUH"
n = 4

def prepare_alphabet(keyword):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    reversed_alphabet = alphabet[::-1]
    cleaned_keyword = "".join(dict.fromkeys(keyword))
    substitution_alphabet = cleaned_keyword + "".join([char for char in alphabet if char not in cleaned_keyword])
    return alphabet, reversed_alphabet,substitution_alphabet

def encrypt(plaintext):
    alphabet, reversed_alphabet, substitution_alphabet = prepare_alphabet(keyword)
    ciphertext = []
    for char in plaintext:
        if char.isalpha(): 
            reverse_char=reversed_alphabet[alphabet.index(char)]
            index = (ord(reverse_char) - ord('A') + n) % 26
            encrypted_char = substitution_alphabet[index]
            ciphertext.append(encrypted_char)
        else:
            ciphertext.append(char)  
    return "".join(ciphertext)

def decrypt(ciphertext):
    alphabet, reversed_alphabet, substitution_alphabet = prepare_alphabet(keyword)
    plaintext = []
    for char in ciphertext:
        if char.isalpha(): 
            index = substitution_alphabet.index(char)
            decrypted_index = (index - n) % 26
            reverse_char = chr(ord('A') + decrypted_index)
            decrypted_char = alphabet[reversed_alphabet.index(reverse_char)]
            plaintext.append(decrypted_char)
        else:
            plaintext.append(char)  
    return "".join(plaintext)