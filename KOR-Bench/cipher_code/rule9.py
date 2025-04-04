class AlbertiCipher:
    def __init__(self, period, increment):
        self.outer_disk = "QWERTYUIOPASDFGHJZXCVBNMKL"
        self.inner_disk = "JKLZXCVBNMASDFGHJQWERTYUIO"
        self.initial_offset = 0
        self.period = period
        self.increment = increment
        self.reset_disks()

    def reset_disks(self):
        self.current_inner_disk = self.inner_disk[self.initial_offset:] + self.inner_disk[:self.initial_offset]

    def encrypt_char(self, char):
        if char in self.outer_disk:
            index = self.outer_disk.index(char)
            return self.current_inner_disk[index]
        else:
            return char

    def decrypt_char(self, char):
        if char in self.current_inner_disk:
            index = self.current_inner_disk.index(char)
            return self.outer_disk[index]
        else:
            return char

    def rotate_disk(self, increment):
        self.current_inner_disk = self.current_inner_disk[increment:] + self.current_inner_disk[:increment]

    def encrypt(self, plaintext):
        self.reset_disks()
        ciphertext = []
        for i, char in enumerate(plaintext):
            ciphertext.append(self.encrypt_char(char))
            if (i + 1) % self.period == 0:
                self.rotate_disk(self.increment)
        return ''.join(ciphertext)

    def decrypt(self, ciphertext):
        self.reset_disks()
        plaintext = []
        for i, char in enumerate(ciphertext):
            plaintext.append(self.decrypt_char(char))
            if (i + 1) % self.period == 0:
                self.rotate_disk(self.increment)
        return ''.join(plaintext)

