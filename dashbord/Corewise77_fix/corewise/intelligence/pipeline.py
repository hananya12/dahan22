

from .identity_engine import IdentityEngine
from .memory_engine import MemoryEngine
from .reid_engine import ReIDEngine



class CorewisePipeline:


    def __init__(self):

        self.identity=IdentityEngine()
        self.memory=MemoryEngine()
        self.reid=ReIDEngine()



    def process(self,tracks):


        output=[]


        for track_id in tracks:


            if track_id not in self.identity.people:

                person=self.identity.create(
                    track_id
                )


            else:

                person=self.identity.people[track_id]



            self.memory.update(
                person["global_id"]
            )


            output.append(person)



        return output

