import math

p= 13
q= 23
    
def is_prime(num) :
    if num == 1 :
        return False
    if num == 2 :
        return True
    if num % 2 == 0 :
        return False

    n = math.floor(math.sqrt(num))
    for i in range(2, n + 1) :
        if num % i == 0 :
            return False
    return True


def gcd_of_two_numbers(a, b) :
    a, b = abs(a), abs(b)
    if a < b :
        a, b = b, a

    remainder = gcd = b
    while (remainder != 0) :
        gcd = remainder
        remainder = a % b
        a = b
        b = remainder
    return gcd


def relatively_prime_or_not(a, b) :
    gcd = gcd_of_two_numbers(a, b)
    if gcd == 1 :
        return True
    else :
        return False


def get_multiplicative_inverse(a, m) :
    gcd = gcd_of_two_numbers(a, m)
    if gcd != 1 :
        return -1
    else :
        i = 0
        ans = 0
        while ans != 1 :
            i += 1
            ans = (a * i) % m
        return i


def rsa_encryption(plaintext) :
    m = (p - 1) * (q - 1)
    n = p * q
    e = m - 1
    while not relatively_prime_or_not(e, m) :
        e = e - 1
    ciphertext = []
    for i in plaintext :
        P = ord(i)
        ciphertext.append((P ** e) % n)
    ciphertext_str = ",".join(str(num) for num in ciphertext)
    return ciphertext_str


def rsa_decryption(ciphertext) :
    ciphertext = [int(num) for num in ciphertext.split(",")]
    m = (p - 1) * (q - 1)
    n = p * q
    e = m - 1
    while not relatively_prime_or_not(e, m) :
        e = e - 1
    d = get_multiplicative_inverse(e, m)
    plaintext = []
    for i in ciphertext :
        plaintext.append(chr((i ** d) % n))
    plaintext = ''.join(plaintext)
    return plaintext


