

class IdentityEngine:

    def __init__(self):
        self.people={}


    def create(self, track_id):

        gid=f"CW_{track_id:04d}"

        self.people[track_id]={
            "global_id":gid
        }

        return self.people[track_id]

