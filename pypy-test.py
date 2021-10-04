import os
import time
import random



start= time.time()






mylist= [ random.randint(1,1000) for i in range(100000000)]


mcount=0
def static(x):
    global  mcount
    if x> 100:
        mcount+=1

mylist=list(map(static,mylist))
end=time.time()

print(mcount)
duration=end-start
print(duration,"(S)")

