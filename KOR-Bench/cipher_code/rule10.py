class JeffersonCipher:
    def __init__(self):
        self.wheel_configuration = [
            "ABCEIGDJFVUYMHTQKZOLRXSPWN",
            "ACDEHFIJKTLMOUVYGZNPQXRWSB",
            "ADKOMJUBGEPHSCZINXFYQRTVWL",
            "AEDCBIFGJHLKMRUOQVPTNWYXZS",
            "AFNQUKDOPITJBRHCYSLWEMZVXG",
            "AGPOCIXLURNDYZHWBJSQFKVMET",
            "AHXJEZBNIKPVROGSYDULCFMQTW",
            "AIHPJOBWKCVFZLQERYNSUMGTDX",
            "AJDSKQOIVTZEFHGYUNLPMBXWCR",
            "AKELBDFJGHONMTPRQSVZUXYWIC",
            "ALTMSXVQPNOHUWDIZYCGKRFBEJ",
            "AMNFLHQGCUJTBYPZKXISRDVEWO",
            "ANCJILDHBMKGXUZTSWQYVORPFE",
            "AODWPKJVIUQHZCTXBLEGNYRSMF",
            "APBVHIYKSGUENTCXOWFQDRLJZM",
            "AQJNUBTGIMWZRVLXCSHDEOKFPY",
            "ARMYOFTHEUSZJXDPCWGQIBKLNV",
            "ASDMCNEQBOZPLGVJRKYTFUIWXH",
            "ATOJYLFXNGWHVCMIRBSEKUPDZQ",
            "AUTRZXQLYIOVBPESNHJWMDGFCK",
            "AVNKHRGOXEYBFSJMUDQCLZWTIP",
            "AWVSFDLIEBHKNRJQZGMXPUCOTY",
            "AXKWREVDTUFOYHMLSIQNJCPGBZ",
            "AYJPXMVKBQWUGLOSTECHNZFRID",
            "AZDNBUHYFWJLVGRCQMPSOEXTKI"
        ]
    
    def encrypt(self, message):
        encrypted_message = []
        wheel_position = 0
        
        for char in message:
            current_wheel = self.wheel_configuration[wheel_position]
            index = current_wheel.index(char)
            encrypted_char = current_wheel[(index + 1) % 26]
            encrypted_message.append(encrypted_char)
            wheel_position = (wheel_position + 1) % len(self.wheel_configuration)
        
        return ''.join(encrypted_message)
    
    def decrypt(self, encrypted_message):
        decrypted_message = []
        wheel_position = 0
        
        for char in encrypted_message:
            current_wheel = self.wheel_configuration[wheel_position]
            index = current_wheel.index(char)
            decrypted_char = current_wheel[(index - 1) % 26]
            decrypted_message.append(decrypted_char)
            wheel_position = (wheel_position + 1) % len(self.wheel_configuration)
        
        return ''.join(decrypted_message)

