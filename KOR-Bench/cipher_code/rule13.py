secretKey="UBILANT"

def checkRepeat(secretKey):
    nonRepeated = ""
    for char in secretKey:
        if nonRepeated == "":
            nonRepeated += char
            continue
        if char in nonRepeated:
            continue
        else:
            nonRepeated += char
    return(nonRepeated)
        

def fillMatrix(nonRepeated):
    nonRepeated.replace("J", "I")
    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ" 
    row = 0
    col = 0
    matrix = [[0 for i in range(5)] for j in range(5)]
    for letter in nonRepeated:
        if col == 5:
            row += 1
            col = 0
        matrix[row][col] = letter
        col += 1
    for letter in alphabet:
        if col == 5:
            row += 1
            col = 0
        if letter in nonRepeated:
            continue
        else:
            matrix[row][col] = letter
            col += 1
    return(matrix)


def cypher(message):
    matrix = fillMatrix(checkRepeat(secretKey))
    message = message.replace('J', '')
    finalCoords = getCoords(message, matrix)
    cyphered = getNewMessage(finalCoords, matrix)
    return cyphered


def getNewMessage(finalCoords, matrix):
    newMessage = ""
    length = len(finalCoords)
    finalCoordsCopy = finalCoords
    for i in range(int(length / 2) + 1):
        x = finalCoordsCopy.pop(0)
        y = finalCoordsCopy.pop(0)
        newMessage += matrix[x][y]
        if not finalCoordsCopy: break

    return(newMessage)
    
def getOldMessage(fullOldCoords, matrix):
    oldMessage = ""
    length = len(fullOldCoords)
    fullOldCoordsCopy = fullOldCoords

    for i in range(int(length / 2) + 1):
        x = int(fullOldCoordsCopy.pop(0))
        y = int(fullOldCoordsCopy.pop(0))
        oldMessage += matrix[x][y]
        if not fullOldCoordsCopy: break
    return(oldMessage)


def getCoords(message, matrix):
    coordsRow = list()
    coordsCol = list()
    for letters in message:
        for i in range(5):
            for j in range (5):
                if matrix[i][j] == letters:
                    coordsRow.append(i)
                    coordsCol.append(j)
                    continue
    finalCoords = list()
    for letters in coordsRow:
        finalCoords.append(letters)
    
    for letters in coordsCol:
        finalCoords.append(letters)
    return(finalCoords)


def decypher(cyphered):
    matrix = fillMatrix(checkRepeat(secretKey))
    return getOldMessage(getOldCoords(getCoords(cyphered, matrix)), matrix)
    

def getOldCoords(coords):
    oldCoords = ''.join(str(coords))
    firstHalf = oldCoords[:len(oldCoords)//2]
    secondHalf = oldCoords[len(oldCoords)//2:]
    fH = "".join(c for c in firstHalf if c.isdecimal())
    sH = "".join(c for c in secondHalf if c.isdecimal())

    firstHalf = list(fH)
    secondHalf = list(sH)

    stepOneCoords = list()
    
    for coord in fH:
        x = firstHalf.pop(0)
        y = secondHalf.pop(0)
        stepOneCoords.append(x)
        stepOneCoords.append(y)

    stepTwoCoords = ''.join(str(stepOneCoords))
    firstHalf = stepTwoCoords[:len(stepTwoCoords)//2]
    secondHalf = stepTwoCoords[len(stepTwoCoords)//2:]
    
    fH = "".join(c for c in firstHalf if c.isdecimal())
    sH = "".join(c for c in secondHalf if c.isdecimal())

    firstHalf = list(fH)
    secondHalf = list(sH)

    fullOldCoords = list()
    
    for coord in fH:
        x = firstHalf.pop(0)
        y = secondHalf.pop(0)
        fullOldCoords.append(x)
        fullOldCoords.append(y)

    return fullOldCoords