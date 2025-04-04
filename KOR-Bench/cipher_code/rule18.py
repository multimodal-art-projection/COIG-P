import numpy as np

template = np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 1, 1]
], dtype=bool)

def create_grid(template,message_segment):
    size = template.shape[0]
    grid = np.full((size, size), '', dtype=str)
    idx = 0
    for _ in range(4):
        for i in range(size):
            for j in range(size):
                if (template[i, j]==0) and idx < len(message_segment):
                    grid[i, j] = message_segment[idx]
                    idx += 1
        template = np.rot90(template)
    return grid

def encrypt(message):
    template = np.array([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1]
    ], dtype=bool)
    
    size = template.shape[0]
    segment_length = size * size
    ciphertext = ''
    for i in range(0, len(message), segment_length):
        segment = message[i:i + segment_length]
        if len(segment) < segment_length:
            segment += '#' * (segment_length - len(segment))  
        filled_grid = create_grid(template, segment)
        ciphertext += ''.join(filled_grid.flatten())
    return ciphertext

def decrypt(ciphertext):
    template = np.array([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1]
    ], dtype=bool)

    size = template.shape[0]
    segment_length = size * size
    message = ''
    for i in range(0, len(ciphertext), segment_length):
        filled_grid = np.array(list(ciphertext[i:i + segment_length])).reshape((size, size))
        segment = []
        for _ in range(4):
            for i in range(size):
                for j in range(size):
                    if template[i, j]==0:
                        segment.append(filled_grid[i, j])
            template = np.rot90(template)
        message += ''.join(segment)
    return message.rstrip('#') 
