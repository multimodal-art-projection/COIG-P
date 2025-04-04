def encrypt_path_write(plain_text):
    text_length = len(plain_text)
    cols = 5
    rows = (text_length + cols - 1) // cols  
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]
    index = 0
    for i in range(rows):
        if i % 2 == 0:
            for j in range(cols):
                if index < len(plain_text):
                    grid[i][j] = plain_text[index]
                    index += 1
        else:
            for j in range(cols - 1, -1, -1):
                if index < len(plain_text):
                    grid[i][j] = plain_text[index]
                    index += 1
    cipher_text = []
    for j in range(cols):
        for i in range(rows):
            char = grid[i][j]
            if (i == rows - 1) and char != ' ':
                cipher_text.append(char)
                cipher_text.append("#")
                break
            elif char == ' ':
                cipher_text.append("#")
                break
            else:
                cipher_text.append(char)
    return ''.join(cipher_text)

def decrypt_path_write(cipher_text):
    cols = 5
    parts = cipher_text.split('#')
    rows = max(len(part) for part in parts)
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]
    for j in range(cols):
        for i in range(len(parts[j])):
                grid[i][j] = parts[j][i]
            
    plain_text = []
    for i in range(rows):
        for j in range(cols):
            if i % 2 == 0:
                plain_text.append(grid[i][j])
            else:
                plain_text.append(grid[i][cols - 1 - j])
    return ''.join(plain_text).replace(' ', '')  




