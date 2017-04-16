import os
import sys
import pickle
import numpy as np
import progressbar

def Create_Mapping_Customer(max_customer, max_movie):
	dirName = 'Netflix/training_set/'
	filenames = os.listdir(dirName)

	MapCustomerIdToIndex = {}
	MapMovieIdToIndex = {}

	indexCustomer = 0
	indexMovie = 0

	flagCust = 1
	for filename in filenames:	
		fp = open(dirName + filename)
		line = fp.readline()
		
		MovieId = int(line.strip(':\n'))
		MapMovieIdToIndex[MovieId] = indexMovie
		indexMovie += 1

		if indexMovie >= max_movie:
			break

		line = fp.readline()
		while line:
			row = line.strip('\n')
			CustId_Rating = row.split(',')
			CustId = int(CustId_Rating[0])
			
			if CustId not in MapCustomerIdToIndex and flagCust == 1:
				MapCustomerIdToIndex[CustId] = indexCustomer
				indexCustomer += 1

				if indexCustomer >= max_customer:
					flagCust = 0

			line = fp.readline()

	print indexMovie
	print indexCustomer
	pickle.dump( MapCustomerIdToIndex, open( "CustId_Index.p", "wb" ) )
	pickle.dump( MapMovieIdToIndex, open( "MovieId_Index.p", "wb" ) )
	


def Create_Data_Set(max_customer, max_movie):
	dirName = 'Netflix/training_set/'
	filenames = os.listdir(dirName)

	MapCustomerIdToIndex = pickle.load( open( "CustId_Index.p", "rb" ) )
	MapMovieIdToIndex = pickle.load( open( "MovieId_Index.p", "rb" ) )

	X_train = np.zeros((max_customer, max_movie))
	Save_Training_Data = "Netflix_train"

	indexMovie = 0
	# bar = progressbar.ProgressBar()
	# with progressbar.ProgressBar(max_value=2000) as bar:
	for filename in filenames:			
		fp = open(dirName + filename)
		line = fp.readline()

		MovieId = int(line.strip(':\n'))
		if MovieId not in MapMovieIdToIndex:
			continue

		col_no = MapMovieIdToIndex[MovieId]

		indexMovie += 1
		sys.stdout.write("Percentage of files: %d   \r" % (1.0*indexMovie/20))
		sys.stdout.flush()

		if indexMovie >= max_movie:
			break

		k = 0
	
		line = fp.readline()
		while line:
			row = line.strip('\n')
			CustId_Rating = row.split(',')
			CustId = int(CustId_Rating[0])
			Rating = int(CustId_Rating[1])
			
			if CustId not in MapCustomerIdToIndex:
				line = fp.readline()
				continue

			row_no = MapCustomerIdToIndex[CustId]			
			X_train[row_no][col_no] = Rating			

			k += 1
			line = fp.readline()
			# sys.stdout.write("File progress: %d   \r" % (k))
			# sys.stdout.flush()


	print X_train[0]
	print X_train[1]
	np.save(Save_Training_Data, X_train)


# def GetMiniBatch(i,B):
# 	NumMovies = 17770
# 	NumCust = 480189

# 	X_batch = np.zeros((B, NumMovies))
# 	# MapMovieIdToIndex = pickle.load( open( "MovieId_Index.p", "rb" ) )
# 	MapCustomerIdToIndex = pickle.load( open( "CustId_Index.p", "rb" ) )

# 	dirName = 'Netflix/training_set/'
# 	filenames = os.listdir(dirName)
	
# 	j = 0
# 	while i < NumCust and j < B:	
# 		filename = filenames[i*B+j]
# 		# i += 1
# 		j += 1

# 		fp = open(dirName + filename)
# 		line = fp.readline()
# 		line = fp.readline()
# 		while line:
# 			row = line.strip('\n')
# 			CustId_Rating = row.split(',')
# 			CustId = int(CustId_Rating[0])
# 			Rating = int(CustId_Rating[1])
			
# 			X_batch[j][MapCustomerIdToIndex[CustId]] = Rating

# 			line = fp.readline()		
		

# 	return X_batch


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
		filename = filenames[i*B+j]
		# i += 1
		j += 1

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
		

	return X_batch