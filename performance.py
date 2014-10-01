from os import*
import os
#os.chdir("D:\workspace\GitHub\CIS565CUDA\Project-2\Project2-StreamCompaction\CIS565_2014_Fall_StreamCompaction\x64\Release");
from subprocess import call
blkSize=range(1,11)
dataSize=range(1,9)
for i in range(10):
	blkSize[i]=2**blkSize[i]
for i in range(8):
	dataSize[i]=10**dataSize[i];
#print blkSize
#print dataSize
#ss=' '.join([".\CIS565_2014_Fall_StreamCompaction.exe", "-b", str(blkSize[0]),"-p",str(dataSize[0]),"-s", str(dataSize[0])])
#print ss
#os.system(' '.join([".\CIS565_2014_Fall_StreamCompaction.exe"]));
for blk in blkSize:
	for ds in dataSize:
		print "Current: "+str(blk)+" "+str(ds)
		os.system(' '.join([".\CIS565_2014_Fall_StreamCompaction.exe", "-b", str(blk),"-p",str(ds),"-s", str(ds)]))
		