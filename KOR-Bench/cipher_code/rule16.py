key = "ABCDE"

def encrypt(plaintext):
    textlen=len(plaintext)
    rows= len(key)
    matrica=[['' for _ in range(textlen)]for _ in range(rows)]
    dict={}
    direction=True
    finish=False
    
    index=0
    cindex=0
    matrica[0][index]=plaintext[cindex]
    cindex+=1
    while(cindex<len(plaintext)):
        if direction:
            for i in range(1,rows):
                if i==1 and cindex!=1:
                    matrica[i-1][index] = '#'
                if cindex == len(plaintext) - 1:
                    matrica[i][index] = plaintext[cindex]
                    finish=True
                    break
                elif i==rows-1:
                    matrica[i][index] = plaintext[cindex]
                    cindex += 1
                    index+=1
                    direction=False
                
                else:
                    matrica[i][index] = plaintext[cindex]
                    cindex += 1
        else:
            for k in range(rows-2,-1,-1):
                if k==rows-2:
                    matrica[k+1][index] = '#'
                if cindex==len(plaintext)-1:
                    matrica[k][index] = plaintext[cindex]
                    finish=True
                    break
                
                elif k==0:
                    matrica[k][index] = plaintext[cindex]
                    cindex += 1
                    index += 1
                    direction = True
                
                else:
                    matrica[k][index] = plaintext[cindex]
                    cindex += 1
        if finish:
            break
    index=0
    for char in key:
        dict[char]=matrica[index]
        index+=1
    myKeys = list(dict.keys())
    myKeys.sort()
    sorted_dict = {i: dict[i] for i in myKeys}
    result=""
    for v in sorted_dict.values():
        for i in range(len(v)):
            if i != len(v)-1:
                result+=v[i]
            elif i == len(v)-1:
                result+="*"
                continue
    return result


def decrypt(cipher):
    textlen = len(cipher)
    rows = len(key)
    rail = [['' for _ in range(textlen)] for _ in range(rows)]
    segments = cipher.split('*')
    for i, segment in enumerate(segments):
        for j, char in enumerate(segment):
            rail[i][j] = char
    decrypted_message=[]
    for i in range(textlen):
        if i % 2 == 0:
            for row in range(rows):
                decrypted_message.append(rail[row][i])
        else:
            for row in range(rows - 1, -1, -1):
                decrypted_message.append(rail[row][i])
    decrypted_text = ''.join(char for char in decrypted_message if char != '#')
    return decrypted_text

