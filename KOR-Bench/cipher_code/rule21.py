key = '10101010'

permutation = (2, 0, 3, 1, 4, 6, 5, 7)
inverse_permutation = (1, 3, 0, 2, 4, 6, 5, 7)

def xor(bits1, bits2):
    return ''.join(['1' if b1 != b2 else '0' for b1, b2 in zip(bits1, bits2)])

def permute(bits, perm_table):
    return ''.join([bits[i] for i in perm_table])

def encrypt(plaintext):
    encrypted_bits = []
    for char in plaintext:
        binary_plaintext = format(ord(char), '08b')
        xor_result = xor(binary_plaintext, key * (len(binary_plaintext) // len(key) + 1))[:len(binary_plaintext)]
        encrypted_bits.append(permute(xor_result, permutation))
    encrypted_binary_string = ''.join(encrypted_bits)
    return encrypted_binary_string

def decrypt(ciphertext):
    decrypted_chars = []
    num_chars = len(ciphertext) // 8  
    for i in range(num_chars):
        binary_ciphertext = ciphertext[i*8:(i+1)*8]
        permuted_bits = permute(binary_ciphertext, inverse_permutation)
        xor_result = xor(permuted_bits, key * (len(permuted_bits) // len(key) + 1))[:len(permuted_bits)]
        decrypted_char = chr(int(xor_result, 2))
        decrypted_chars.append(decrypted_char)
    return ''.join(decrypted_chars)


