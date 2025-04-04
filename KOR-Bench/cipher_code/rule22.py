S_BOX = {
    0x00: 0x0F, 0x01: 0x0A, 0x02: 0x07, 0x03: 0x05,
    0x04: 0x09, 0x05: 0x03, 0x06: 0x0D, 0x07: 0x00,
    0x08: 0x0E, 0x09: 0x08, 0x0A: 0x04, 0x0B: 0x06,
    0x0C: 0x01, 0x0D: 0x02, 0x0E: 0x0B, 0x0F: 0x0C
}

INV_S_BOX = {
    0x0F: 0x00, 0x0A: 0x01, 0x07: 0x02, 0x05: 0x03,
    0x09: 0x04, 0x03: 0x05, 0x0D: 0x06, 0x00: 0x07,
    0x0E: 0x08, 0x08: 0x09, 0x04: 0x0A, 0x06: 0x0B,
    0x01: 0x0C, 0x02: 0x0D, 0x0B: 0x0E, 0x0C: 0x0F
}

KEY = b'1234567890ABCDEF'

def xor_bytes(a, b):
    return bytes(x ^ y for x, y in zip(a, b))

def substitute(bytes_data, box):
    result = bytearray()
    for byte in bytes_data:
        high_nibble = (byte >> 4) & 0x0F
        low_nibble = byte & 0x0F
        substituted_high = box[high_nibble]
        substituted_low = box[low_nibble]
        combined_byte = (substituted_high << 4) | substituted_low
        if combined_byte < 0 or combined_byte > 255:
            raise ValueError(f"Invalid byte value: {combined_byte}")
        result.append(combined_byte)
    return bytes(result)

def simple_permute(bytes_data):
    return bytes(((b << 1) & 0xFF) | ((b >> 7) & 0x01) for b in bytes_data)

def inverse_permute(bytes_data):
    return bytes(((b >> 1) & 0xFF) | ((b << 7) & 0x80) for b in bytes_data)

def encrypt(plaintext):
    blocks = [plaintext[i:i+8] for i in range(0, len(plaintext), 8)]
    encrypted_blocks = []
    for block in blocks:
        block = block.ljust(8, '\x00')  
        state = xor_bytes(block.encode('ascii'), KEY)
        state = substitute(state, S_BOX)
        state = simple_permute(state)
        state = xor_bytes(state, KEY)
        encrypted_blocks.append(''.join(f'{b:02X}' for b in state))
    return ''.join(encrypted_blocks)

def decrypt(ciphertext):
    blocks = [ciphertext[i:i+16] for i in range(0, len(ciphertext), 16)]
    decrypted_blocks = []
    for block in blocks:
        state = bytes.fromhex(block)
        state = xor_bytes(state, KEY)
        state = inverse_permute(state)
        state = substitute(state, INV_S_BOX)
        state = xor_bytes(state, KEY)
        decrypted_blocks.append(state.decode('ascii').rstrip('\x00'))
    return ''.join(decrypted_blocks)
