import os
import sys
import pickle

# dirName = 'Netflix/training_set/'
# filenames = os.listdir(dirName)

# MapMovieIdToIndex = {}
# MapCustomerIdToIndex = {}
# indexMovie = 0
# indexCustomer = 0

# for filename in filenames:	
# 	fp = open(dirName + filename)
# 	line = fp.readline()

# 	MovieId = int(line.strip(':\n'))
# 	MapMovieIdToIndex[MovieId] = indexMovie
# 	indexMovie += 1

# 	sys.stdout.write("Creating Mapping progress: %d   \r" % (indexMovie))
# 	sys.stdout.flush()

# 	line = fp.readline()
# 	while line:
# 		row = line.strip('\n')
# 		CustId_Rating = row.split(',')
# 		CustId = int(CustId_Rating[0])
		
# 		if CustId not in MapCustomerIdToIndex:
# 			MapCustomerIdToIndex[CustId] = indexCustomer
# 			indexCustomer += 1

# 		line = fp.readline()

# # print indexMovie
# print indexCustomer

# pickle.dump( MapCustomerIdToIndex, open( "CustId_Index.p", "wb" ) )
# pickle.dump( MapMovieIdToIndex, open( "MovieId_Index.p", "wb" ) )

def GetMiniBatch(i,B):
	NumMovies = 17770
	NumCust = 480189

	X_batch = np.zeros((B, NumCust))
	# MapMovieIdToIndex = pickle.load( open( "MovieId_Index.p", "rb" ) )
	MapCustomerIdToIndex = pickle.load( open( "CustId_Index.p", "rb" ) )

	dirName = 'Netflix/training_set/'
	filenames = os.listdir(dirName)
	
	j = 0
	while i < NumMovies and j < B:	
		filename = filenames[i]
		i += 1
		
		fp = open(dirName + filename)
		line = fp.readline()
		line = fp.readline()
		while line:
			row = line.strip('\n')
			CustId_Rating = row.split(',')
			CustId = int(CustId_Rating[0])
			Rating = int(CustId_Rating[1])
			
			X_batch[j][MapCustomerIdToIndex[CustId]] = Rating

			line = fp.readline()
		
		j += 1

	return X_batch