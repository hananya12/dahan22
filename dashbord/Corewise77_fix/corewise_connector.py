

from corewise.intelligence.pipeline import CorewisePipeline



corewise_engine = CorewisePipeline()



def process_corewise(results):


    ids=[]


    if results.boxes.id is not None:

        ids=[
            int(x)
            for x in results.boxes.id.cpu().numpy()
        ]


    return corewise_engine.process(ids)

