from random import randint

ALPHABET_CZ = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
ALPHABET_EN = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
ALPHABET_ALL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

key="ABCDEF"
class Matrix:
    def __init__(self, matrixType=""):
        self.matrix = []
        self.setType(matrixType)

    def setType(self, matrixType):
        self.type = matrixType
        if self.type == "cz":
            self.length = 5
            self.alphabet = ALPHABET_CZ
        elif self.type == "en":
            self.length = 5
            self.alphabet = ALPHABET_EN
        else:
            self.length = 6
            self.alphabet = ALPHABET_ALL
        self.clean()
        self.fill()

    def fill(self):
        remains = self.alphabet
        for i in range(self.length):
            for j in range(self.length):
                self.matrix[i][j] = remains[randint(0, len(remains) - 1)]
                remains = remains.replace(self.matrix[i][j], '')

    def clean(self):
        self.matrix = [['' for i in range(self.length)] for j in range(self.length)]

    def find(self, letter):
        for i in range(self.length):
            for j in range(self.length):
                if self.matrix[i][j] == letter:
                    return (i, j)
matrix = Matrix()
matrix.matrix =[['R', 'U', 'A', '0', 'Q', 'B'], ['D', '2', 'W', 'K', 'S', '1'], ['H', '4', '5', 'F', 'T', 'Z'], ['Y', 'C', 'G', 'X', '7', 'L'], ['9', '8', 'I', '3', 'P', 'N'], ['6', 'J', 'V', 'O', 'E', 'M']]


def encrypt(text):
    line = "ADFGVX" if matrix.length == 6 else "ADFGX"
    encrypted = ""
    for i in text:
        row, col = matrix.find(i)
        encrypted += line[row]
    for i in text:
        row, col = matrix.find(i)
        encrypted += line[col]
    return encrypted


def decrypt(text):
    line = "ADFGVX" if matrix.length == 6 else "ADFGX"
    decryptedText = ""
    half_length = len(text) // 2
    first_half = text[:half_length]
    second_half = text[half_length:]
    for pair in zip(first_half, second_half):
        decryptedText += matrix.matrix[line.index(pair[0])][line.index(pair[1])]
    return decryptedText
