

class ReIDEngine:


    def __init__(self):

        self.database={}



    def save(self,gid,data):

        self.database[gid]=data



    def match(self,data):

        return None,0

