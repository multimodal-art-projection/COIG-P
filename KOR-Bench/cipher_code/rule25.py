import hashlib

def generate_key():
    secret = "SECRET_KEY"
    sha256 = hashlib.sha256()
    sha256.update(secret.encode('utf-8'))
    return sha256.digest()

def encrypt(plaintext):
    key = generate_key()
    print(key)
    hex_string = key.hex()
    print(hex_string)
    plaintext_bytes = plaintext.encode('utf-8')
    ciphertext_bytes = bytes([b ^ key[i % len(key)] for i, b in enumerate(plaintext_bytes)])
    return ciphertext_bytes.hex()

def decrypt(ciphertext):
    key = generate_key()
    ciphertext_bytes = bytes.fromhex(ciphertext)
    plaintext_bytes = bytes([b ^ key[i % len(key)] for i, b in enumerate(ciphertext_bytes)])
    return plaintext_bytes.decode('utf-8')