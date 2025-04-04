alphabet = "XMJQUDONPRGTVBWFAKSHZCYEIL"
A = 3
B = 5

def affine_encrypt(message):
    encrypted_message = []
    n = len(alphabet)
    letter_to_index = {char: idx for idx, char in enumerate(alphabet)}
    
    for char in message:
        if char in letter_to_index:
            x = letter_to_index[char]  
            y = (A * x + B) % n        
            encrypted_message.append(alphabet[y])
        else:
            encrypted_message.append(char)  
    
    return ''.join(encrypted_message)

def mod_inverse(A, m):
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while A > 1:
        q = A // m
        m, A = A % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

def affine_decrypt(ciphertext):
    decrypted_message = []
    n = len(alphabet)
    letter_to_index = {char: idx for idx, char in enumerate(alphabet)}
    A_inv = mod_inverse(A, n) 
    
    for char in ciphertext:
        if char in letter_to_index:
            y = letter_to_index[char]  
            x = A_inv * (y - B) % n     
            decrypted_message.append(alphabet[x])
        else:
            decrypted_message.append(char)  
    
    return ''.join(decrypted_message)



