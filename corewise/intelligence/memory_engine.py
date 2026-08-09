

import time


class MemoryEngine:

    def __init__(self):
        self.memory={}


    def update(self,gid):

        self.memory[gid]={
            "last_seen":time.time()
        }


