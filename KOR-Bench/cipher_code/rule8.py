class PortaCipher:
    def __init__(self, key):
        self.key = key.upper()
        self.alphabets = [
            'NOPQRSTUVWXYZABCDEFGHIJKLM',
            'ZNOPQRSTUVWXYBCDEFGHIJKLMA',
            'YZNOPQRSTUVWXCDEFGHIJKLMAB',
            'XYZNOPQRSTUVWDEFGHIJKLMABC',
            'WXYZNOPQRSTUVEFGHIJKLMABCD',
            'VWXYZNOPQRSTUFGHIJKLMABCDE',
            'UVWXYZNOPQRSTGHIJKLMABCDEF',
            'TUVWXYZNOPQRSHIJKLMABCDEFG',
            'STUVWXYZNOPQRIJKLMABCDEFGH',
            'RSTUVWXYZNOPQJKLMABCDEFGHI',
            'QRSTUVWXYZNOPKLMABCDEFGHIJ',
            'PQRSTUVWXYZNOLMABCDEFGHIJK',
            'OPQRSTUVWXYZNMABCDEFGHIJKL'
        ]
        self.char_to_alphabet_index = {chr(i + ord('A')): i // 2 for i in range(26)}

    def encrypt_char(self, char, key_char):
        index = self.char_to_alphabet_index[key_char]
        return self.alphabets[index][ord(char) - ord('A')]

    def decrypt_char(self, char, key_char):
        index = self.char_to_alphabet_index[key_char]
        return chr(self.alphabets[index].index(char) + ord('A'))

    def encrypt(self, plaintext):
        plaintext = plaintext.upper()
        ciphertext = []
        for i, char in enumerate(plaintext):
            if char.isalpha():
                key_char = self.key[i % len(self.key)]
                ciphertext.append(self.encrypt_char(char, key_char))
            else:
                ciphertext.append(char)
        return ''.join(ciphertext)

    def decrypt(self, ciphertext):
        ciphertext = ciphertext.upper()
        plaintext = []
        for i, char in enumerate(ciphertext):
            if char.isalpha():
                key_char = self.key[i % len(self.key)]
                plaintext.append(self.decrypt_char(char, key_char))
            else:
                plaintext.append(char)
        return ''.join(plaintext)
