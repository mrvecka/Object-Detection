import numpy as np 
import config as cfg

class LRQueue(): 
    
    def __init__(self):
        self.items = []
        self.stop = False
        
        
    def put(self, item):
        try:
            val = float(item)
            if len(self.items) == 10:
                # remove first element and insert to the end
                self.items.pop(0)
                self.items.append(val)
            else:
                self.items.append(val)

            self.checkError()                
        except ValueError:
            print("Queue is type of int, other values will be ignored!")
            
    def checkError(self):
        avg = np.sum(self.items) / 10.0
        if avg < cfg.MAX_ERROR:
            self.stop = True
            
    def stop_training(self):
        return self.stop
        
        