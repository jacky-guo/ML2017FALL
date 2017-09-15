
import sys

if __name__ == "__main__":
	txt = sys.argv[1]
	f = open(txt,'r',encoding='UTF-8')
	wordDict = {}
	for i in open(txt):
		for j in i.split():
			if j in wordDict.keys():
				wordDict[j] += 1
			else :
				wordDict[j] = 1

	f2 = open('Q1.txt','w',encoding='UTF-8')
	for i,j in enumerate (wordDict.items()):
		if j != list(wordDict.items())[-1]:
			f2.write(j[0]+' '+str(i)+' '+str(j[1])+'\n')
		else:
			f2.write(j[0]+' '+str(i)+' '+str(j[1]))			

	f.close()
	f2.close()