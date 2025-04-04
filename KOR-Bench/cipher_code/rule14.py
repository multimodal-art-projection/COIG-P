import numpy as np

grid1 = np.array([
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O"],
    ["P", "A", "S", "D", "F", "G", "H", "J", "K"],
    ["L", "Z", "X", "C", "V", "B", "N", "M", "#"]
])

grid2 = np.array([
    ["Q", "W", "E"],
    ["R", "T", "Y"], 
    ["U", "I", "O"],
    ["P", "A", "S"], 
    ["D", "F", "G"], 
    ["H", "J", "K"],
    ["L", "Z", "X"], 
    ["C", "V", "B"], 
    ["N", "M", "#"]
])


grid3 = np.array([
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9]
])

def find_position(grid, char):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == char:
                return (i, j)
    return None

def encrypt_pair(l1, l2):
    l1_row, l1_col = find_position(grid1, l1)
    l2_row, l2_col = find_position(grid2, l2)
    num3 = grid3[l1_row, l2_col]
    col1 = l1_col
    row2 = l2_row

    return col1, num3, row2

def decrypt_triple(x, y, z):
    l1_col = x
    l2_row = z
    l1_row, l2_col = find_position(grid3, y)
    l1 = grid1[l1_row, l1_col]
    l2 = grid2[l2_row, l2_col]

    return l1, l2

def encrypt_message(message):
    while len(message) % 6 != 0:
        message += "#"
    bigrams = [message[i:i+2] for i in range(0, len(message), 2)]
    triples = [encrypt_pair(l1, l2) for l1, l2 in bigrams]
    print(triples)
    encrypted_pairs = ["".join(map(str, triple)) for triple in triples]
    encrypted_message = "".join(encrypted_pairs)
    return encrypted_message

def chunk_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def decrypt_message(encrypted_message):
    original_triples = chunk_list(encrypted_message, 3)
    original_triples =  [[int(item) for item in row] for row in original_triples]
    decrypted_pairs = [decrypt_triple(*triple) for triple in original_triples]
    decrypted_message = "".join(sum(decrypted_pairs, ()))
    decrypted_message = decrypted_message.replace("#", "")
    return decrypted_message

