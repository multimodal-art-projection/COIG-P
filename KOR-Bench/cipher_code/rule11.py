import re

key = ['ECHO', 'VORTEX']

table = [['K', 'L', 'M', 'N', 'O'],
        ['P', 'R', 'S', 'T', 'U'],
        ['V', 'W', 'X', 'Y', 'Z'],
        ['A', 'B', 'C', 'D', 'E'],
        ['F', 'G', 'H', 'I', 'J']]

def generate_table(key = ''):	
    alphabet = 'ABCDEFGHIJKLMNOPRSTUVWXYZ'	
    table = [[0] * 5 for row in range(5)] 

    key = re.sub(r'[\W]', '', key).upper()
    for row in range(5):
        for col in range(5):
            if len(key) > 0:
                table[row][col] = key[0]                
                alphabet = alphabet.replace(key[0], '') 
                key = key.replace(key[0], '')           
            else:
                table[row][col] = alphabet[0]
                alphabet = alphabet[1:]                 
    
    return table

def encrypt(plaintext):
    ciphertext = ''
    plaintext = re.sub(r'[\W]', '', plaintext).upper().replace('Q', '')
    if (len(plaintext)%2!=0):
        plaintext = plaintext + 'X'
    topRight, bottomLeft  = generate_table(key[0]), generate_table(key[1])      

    for i in range(0, len(plaintext), 2):   
        digraphs = plaintext[i:i+2]         
        ciphertext += mangle(topRight, bottomLeft, digraphs)
    return ciphertext

def mangle(topRight, bottomLeft, digraphs):
    a = position(table, digraphs[0])        
    b = position(table, digraphs[1])
    return topRight[a[0]][a[1]] + bottomLeft[b[0]][b[1]]    

def decrypt(ciphertext):	
    plaintext = ''	
    topRight, bottomLeft = generate_table(key[0]), generate_table(key[1])
    for i in range(0, len(ciphertext), 2):
        digraphs = ciphertext[i:i+2]
        plaintext += unmangle(topRight, bottomLeft, digraphs)
    return plaintext

def unmangle(topRight, bottomLeft, digraphs):
    a = position(topRight, digraphs[0])
    b = position(bottomLeft, digraphs[1])
    return table[a[0]][a[1]] + table[b[0]][b[1]]

def position(table, ch):	
    for row in range(5):	
        for col in range(5):	
            if table[row][col] == ch:	
                return (row, col)       
    return (None, None)

