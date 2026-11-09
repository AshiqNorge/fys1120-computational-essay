import time as time
import numpy as plt

class Klokke:
    def __init__(self, N, percentlist = (0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.), step=0):
        if type(N) == "int" or type(N) == "float":
            self.N = int(N)
        else:
            self.N = plt.max(N)
        self.start = time.time()
        self.percentlist = percentlist
        self.step = step

    def timer(self, i):
        tid = (i+1)/self.N
        if tid >= self.percentlist[self.step]:
            elapsedtime = time.time() - self.start
            print(f"Progressbar: {100*tid:.0f}%, approximate time left: {elapsedtime/self.percentlist[self.step]*(1-self.percentlist[self.step]):.1f} seconds")
            self.step += 1
        if tid >= 1:
            end = time.time()
            print(f"Koden kjørte på {(end-self.start):.1f} sekunder")

    def __rsub__(self, other):
        return self.start - other


