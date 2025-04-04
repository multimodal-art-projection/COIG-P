import numpy as np

keyword = "PHILLIPS"

def generate_initial_grid(keyword):
    cleaned_keyword = keyword.upper().replace('J', 'I')
    cleaned_keyword = ''.join(sorted(set(cleaned_keyword), key=cleaned_keyword.index))
    alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXYZ' 
    grid = np.array(list(cleaned_keyword + ''.join(filter(lambda c: c not in cleaned_keyword, alphabet))))
    grid = grid.reshape(5, 5)
    return grid

def generate_subsequent_grids(initial_grid):
    grids = [initial_grid]
    for i in range(1, 5):
        grid = np.roll(initial_grid, i, axis=0)
        grids.append(grid)

    for i in range(1, 4):
        grid = np.roll(grids[4], i, axis=0)
        grids.append(grid)

    return grids

def find_position(grid, letter):
    indices = np.where(grid == letter)
    return indices[0][0], indices[1][0]

def phillips_encrypt(message, initial_grid):
    grids = generate_subsequent_grids(initial_grid)
    encrypted_message = []

    for i in range(0, len(message), 5):
        block = message[i:i+5]
        grid_index = (i // 5) % 8
        grid = grids[grid_index]

        encrypted_block = []

        for letter in block:
            if letter.upper() == 'J':
                encrypted_block.append(letter)  
            else:
                row, col = find_position(grid, letter.upper())
                encrypted_letter = grid[(row + 1) % 5, (col + 1) % 5]
                encrypted_block.append(encrypted_letter)

        encrypted_message.append(''.join(encrypted_block))

    return ''.join(encrypted_message)


def phillips_decrypt(message, initial_grid):
    grids = generate_subsequent_grids(initial_grid)
    decrypted_message = []

    for i in range(0, len(message), 5):
        block = message[i:i+5]
        grid_index = (i // 5) % 8
        grid = grids[grid_index]

        decrypted_block = []

        for letter in block:
            if letter.upper() == 'J':
                decrypted_block.append(letter)  
            else:
                row, col = find_position(grid, letter.upper())
                decrypted_letter = grid[(row - 1) % 5, (col - 1) % 5]
                decrypted_block.append(decrypted_letter)

        decrypted_message.append(''.join(decrypted_block))

    return ''.join(decrypted_message)

def encrypt(message):
    initial_grid = generate_initial_grid(keyword)
    encrypted_message = phillips_encrypt(message, initial_grid)
    return encrypted_message

def decrypt(message):
    initial_grid = generate_initial_grid(keyword)
    decrypted_message = phillips_decrypt(message, initial_grid)
    return decrypted_message

