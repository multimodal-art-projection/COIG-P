polybius_square = [
    ['R', 'T', 'X', 'F', 'S'],
    ['W', 'C', 'M', 'V', 'H'],
    ['Z', 'J', 'A', 'P', 'B'],
    ['L', 'Q', 'Y', 'G', 'K'],
    ['N', 'E', 'U', 'D', 'I']
]

def find_position(char):
    for i in range(len(polybius_square)):
        for j in range(len(polybius_square[i])):
            if polybius_square[i][j] == char:
                return i + 1, j + 1  
    return None

def encrypt(plaintext):
    plaintext = plaintext.upper() 
    encrypted_text = ""
    for char in plaintext:
        if char == 'O':
            encrypted_text += '66'  
        elif char.isalpha():
            row, col = find_position(char)
            encrypted_text += f"{row}{col}"  
        elif char.isspace():
            encrypted_text += ' '  
    return encrypted_text.strip()

def decrypt(encrypted_text):
    decrypted_text = ""
    i = 0
    while i < len(encrypted_text):
        if encrypted_text[i:i+2] == '66':
            decrypted_text += 'O'  
            i += 2  
        elif encrypted_text[i].isdigit():
            row = int(encrypted_text[i])
            col = int(encrypted_text[i+1])
            decrypted_text += polybius_square[row-1][col-1]  
            i += 2  
        elif encrypted_text[i].isspace():
            decrypted_text += ' '  
            i += 1  
    return decrypted_text