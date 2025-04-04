def get_inverse(mu, p):
    for i in range(1, p):
        if (i*mu)%p == 1:
            return i
    return -1

def get_gcd(zi, mu):
    if mu:
        return get_gcd(mu, zi%mu)
    else:
        return zi

def get_np(x1, y1, x2, y2, a, p):
    flag = 1  
    if x1 == x2 and y1 == y2:
        zi = 3 * (x1 ** 2) + a  
        mu = 2 * y1    
    else:
        zi = y2 - y1
        mu = x2 - x1
        if zi* mu < 0:
            flag = 0        
            zi = abs(zi)
            mu = abs(mu)
    gcd_value = get_gcd(zi, mu)     
    zi = zi // gcd_value            
    mu = mu // gcd_value
    inverse_value = get_inverse(mu, p)
    k = (zi * inverse_value)
    if flag == 0:                   
        k = -k
    k = k % p
    x3 = (k ** 2 - x1 - x2) % p
    y3 = (k * (x1 - x3) - y1) % p
    return x3,y3

def get_rank(x0, y0, a, b, p):
    x1 = x0            
    y1 = (-1*y0)%p     
    tempX = x0
    tempY = y0
    n = 1
    while True:
        n += 1
        p_x,p_y = get_np(tempX, tempY, x0, y0, a, p)
        if p_x == x1 and p_y == y1:
            return n+1
        tempX = p_x
        tempY = p_y

def get_param(x0, a, b, p):
    y0 = -1
    for i in range(p):
        if i**2%p == (x0**3 + a*x0 + b)%p:
            y0 = i
            break
    if y0 == -1:
        return False
    x1 = x0
    y1 = (-1*y0) % p
    return x0,y0,x1,y1

def get_graph(a, b, p):
    x_y = []
    for i in range(p):
        x_y.append(['-' for i in range(p)])
    for i in range(p):
        val =get_param(i, a, b, p)  
        if(val != False):
            x0,y0,x1,y1 = val
            x_y[x0][y0] = 1
            x_y[x1][y1] = 1
    for i in range(p):              
        temp = p-1-i        
        if temp >= 10:
            print(temp, end=" ")
        else:
            print(temp, end="  ")
        for j in range(p):
            print(x_y[j][temp], end="  ")
        print("")   
    print("  ", end="")
    for i in range(p):
        if i >=10:
            print(i, end=" ")
        else:
            print(i, end="  ")
    print('\n')

def get_ng(G_x, G_y, key, a, p):
    temp_x = G_x
    temp_y = G_y
    while key != 1:
        temp_x,temp_y = get_np(temp_x,temp_y, G_x, G_y, a, p)
        key -= 1
    return temp_x,temp_y


def encrypt(plaintext):
    a = 13
    b = 19
    p = 17
    G_x = 1
    G_y = 13
    n = get_rank(G_x, G_y, a, b, p)
    print(n)
    key=1
    KEY_x,kEY_y = get_ng(G_x, G_y, key, a, p)
    print(KEY_x,kEY_y)
    k=3
    k_G_x,k_G_y = get_ng(G_x, G_y, k, a, p)                         
    k_Q_x,k_Q_y = get_ng(KEY_x, kEY_y, k, a, p)                     

    print(k_G_x)
    print(k_G_y)
    print(k_Q_x)
    print(k_Q_y)
    

    c = []
    print(end="")
    for char in plaintext:
        intchar = ord(char)
        cipher_text = intchar*k_Q_x
        c.append(cipher_text)
        
    ciphertext_str = ",".join(str(num) for num in c)
    
    return ciphertext_str

def decrypt(ciphertext):
    plain_text=""
    ciphertext = [int(num) for num in ciphertext.split(",")]
    for charArr in ciphertext:
        decrypto_text_x=12
        plain_text+=chr(charArr//decrypto_text_x)
    return plain_text

