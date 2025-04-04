grid = [['M', 'Z', 'S', 'D', 'P'],
        ['K', 'N', 'F', 'L', 'Q'],
        ['G', 'A', 'O', 'X', 'U'],
        ['W', 'R', 'Y', 'V', 'C'], 
        ['B', 'T', 'E', 'H', 'I']
        ]

def collon_encrypt(plaintext):
    rows = len(grid)
    cols = len(grid[0])
    plaintext = plaintext.replace(" ", "").upper()
    plaintext = plaintext.replace("J","")
    encrypted_message = []
    for i in range(0, len(plaintext)):
        char = plaintext[i]
        found = False
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == char:
                    encrypted_message.append(grid[r][0] + grid[cols-1][c])
                    found = True
                    break
            if found:
                break
    encrypted_message = ''.join(encrypted_message)
    return encrypted_message

def collon_decrypt(ciphertext):
    rows = len(grid)
    cols = len(grid[0])
    decrypted_message = []
    for i in range(0, len(ciphertext),2):
        bigram = ciphertext[i:i + 2]
        if bigram == '':
            break
        row_header = bigram[0]
        bottom_of_column = bigram[1]
        found = False
        for r in range(rows):
            if grid[r][0] == row_header:
                for c in range(cols):
                    if grid[cols-1][c] == bottom_of_column:
                        decrypted_message.append(grid[r][c])
                        found = True
                        break
                if found:
                    break
    decrypted_message = ''.join(decrypted_message)
    return decrypted_message