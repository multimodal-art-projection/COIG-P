import secrets

key = "MISSION"
key2 = "SECURE"

def preprocess_forward(plaintext):
    plaintext_stream=[]
    for char in plaintext:
        num=ord(char)-ord("A")
        plaintext_stream.append(num)
    random_stream=[]
    for i in range(len(plaintext_stream)):
        random_num=secrets.randbelow(26)
        random_stream.append(random_num)
    preprocessed_stream=[]
    for i in range(len(plaintext_stream)):
        char_num=plaintext_stream[i]
        random_num=random_stream[i]
        a=((2*char_num)+random_num)%26
        b=(char_num+random_num)%26
        preprocessed_stream.append(a)
        preprocessed_stream.append(b)
    preprocessed_plaintext=""
    for num in preprocessed_stream:
        char=chr(num+ord("A"))
        preprocessed_plaintext=preprocessed_plaintext+char
    return(preprocessed_plaintext)

def preprocess_backward(preprocessed_plaintext):
    number_stream=[]
    for char in preprocessed_plaintext:
        num=ord(char)-ord("A")
        number_stream.append(num)
    tuple_stream=[]
    for i in range(0, len(number_stream), 2):
        tuple_stream.append((number_stream[i],number_stream[i+1]))
    plaintext_stream=[]
    for tup in tuple_stream:
        a=tup[0]
        b=tup[1]
        c=a-b
        if c<0:
            plaintext_num=c+26
        else:
            plaintext_num=c
        plaintext_stream.append(plaintext_num)
    plaintext=""
    for num in plaintext_stream:
        char=chr(num+ord("A"))
        plaintext=plaintext+char
    return(plaintext)

def transpose_encrypt(plaintext):
    columns = len(key)
    rows = (len(plaintext) + columns - 1) // columns
    grid = []
    for i in range(rows):
        a = []
        for j in range(columns):
            a.append("")
        grid.append(a)
    index = 0
    for row in range(rows):
        for col in range(columns):
            if index < len(plaintext):
                grid[row][col] = plaintext[index]
                index=index+1
            else:
                grid[row][col]="$"
    sorted_columns = [col for col in range(columns)]
    sorted_columns.sort(key=lambda x: key[x])

    new_grid=[]
    for i in range(rows):
        a = []
        for j in range(columns):
            a.append("")
        new_grid.append(a)
    for row in range(rows):
        for col in range(columns):
            new_grid[row][col]=grid[row][sorted_columns[col]]
    ciphertext = ""
    for row in range(rows):
        for col in range(columns):
            char=new_grid[row][col]
            ciphertext=ciphertext+char
    return (ciphertext)

def transpose_decrypt(ciphertext):
    columns = len(key)
    rows = (len(ciphertext) + columns - 1) // columns
    grid = []
    for i in range(rows):
        a = []
        for j in range(columns):
            a.append("")
        grid.append(a)
    sorted_columns = [col for col in range(columns)]
    sorted_columns.sort(key=lambda x: key[x])
    index = 0
    for row in range(rows-1):
        for col in range(columns):
            if index < len(ciphertext):
                grid[row][col] = ciphertext[index]
                index=index+1
    a=(len(ciphertext)%columns)
    if a==0:
        for col in range(columns):
            grid[rows-1][col]=ciphertext[index]
            index=index+1
    else:
        for col in range(columns):
            current_collum_index=sorted_columns[col]
            if current_collum_index>=a:
                grid[rows-1][col]="$"
            else:
                grid[rows-1][col]=ciphertext[index]
                index=index+1
    new_grid=[]
    for i in range(rows):
        a = []
        for j in range(columns):
            a.append("")
        new_grid.append(a)
    index=0
    for col in sorted_columns:
        for row in range(rows):
            char=grid[row][index]
            new_grid[row][col]=char
        index=index+1
    plaintext=""
    for row in range(rows):
        for col in range(columns):
            char=new_grid[row][col]
            if char=="$":
                pass   
            else:
                plaintext=plaintext+char
    return(plaintext)