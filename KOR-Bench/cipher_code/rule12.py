class MorseCodeCipher:
    def __init__(self, key):
        self.key = key
        self.index_mapping = self.generate_index_mapping()
        self.reverse_mapping = {v: k for k, v in self.index_mapping.items()}
    
    def generate_index_mapping(self):
        sorted_key = sorted(self.key)
        index_mapping = {
            '..': sorted_key.index(self.key[0]) + 1,
            '.-': sorted_key.index(self.key[1]) + 1,
            './': sorted_key.index(self.key[2]) + 1,
            '-.': sorted_key.index(self.key[3]) + 1,
            '--': sorted_key.index(self.key[4]) + 1,
            '-/': sorted_key.index(self.key[5]) + 1,
            '/.': sorted_key.index(self.key[6]) + 1,
            '/-': sorted_key.index(self.key[7]) + 1,
            '//': sorted_key.index(self.key[8]) + 1,
        }
        return index_mapping
    
    def encrypt(self, plaintext):
        morse_code = self.text_to_morse(plaintext)
        encrypted_numbers = []
        
        pairs = [morse_code[i:i+2] for i in range(0, len(morse_code), 2)]
        for pair in pairs:
            if len(pair) % 2 == 0:
                index = self.index_mapping[pair]
                encrypted_numbers.append(str(index))
            else:
                encrypted_numbers.append(pair)
        
        encrypted_message = ''.join(encrypted_numbers)
        return encrypted_message
    
    def decrypt(self, encrypted_message):
        decrypted_numbers = []
    
        for char in encrypted_message:
            if char.isdigit():
                index = int(char)
                decrypted_numbers.append(self.reverse_mapping[index])
            else:
                decrypted_numbers.append(char)
        
        decrypted_morse = ''.join(decrypted_numbers)
        plaintext = self.morse_to_text(decrypted_morse)
        return plaintext
    
    def text_to_morse(self, text):
        morse_code = {
            'A': '.-',     'B': '-...',   'C': '-.-.',   'D': '-..',
            'E': '.',      'F': '..-.',   'G': '--.',    'H': '....',
            'I': '..',     'J': '.---',   'K': '-.-',    'L': '.-..',
            'M': '--',     'N': '-.',     'O': '---',    'P': '.--.',
            'Q': '--.-',   'R': '.-.',    'S': '...',    'T': '-',
            'U': '..-',    'V': '...-',   'W': '.--',    'X': '-..-',
            'Y': '-.--',   'Z': '--..',
        }
        
        morse_chars = []
        words = text.split(' ')
        
        for word in words:
            chars = []
            for char in word:
                if char.upper() in morse_code:
                    chars.append(morse_code[char.upper()])
            morse_chars.append('/'.join(chars))
        
        return '//'.join(morse_chars)
    
    def morse_to_text(self, morse_code):
        morse_code = morse_code.split('//')
        morse_to_char = {
            '.-': 'A',     '-...': 'B',   '-.-.': 'C',   '-..': 'D',
            '.': 'E',      '..-.': 'F',   '--.': 'G',    '....': 'H',
            '..': 'I',     '.---': 'J',   '-.-': 'K',    '.-..': 'L',
            '--': 'M',     '-.': 'N',     '---': 'O',    '.--.': 'P',
            '--.-': 'Q',   '.-.': 'R',    '...': 'S',    '-': 'T',
            '..-': 'U',    '...-': 'V',   '.--': 'W',    '-..-': 'X',
            '-.--': 'Y',   '--..': 'Z',
        }
        morse_chars = []
        for word in morse_code:
            chars = []
            for char in word.split('/'):
                chars.append(morse_to_char[char])
            morse_chars.append(''.join(chars))
        return ' '.join(morse_chars)
