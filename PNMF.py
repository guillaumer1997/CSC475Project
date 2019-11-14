import numpy as np
		
def PNMF(X, WD, HD, WH, HH, rh, sparsity):
	X = X + np.finfo(np.double).tiny #make sure there's no zero frame

	shapeX = np.shape(X)
	numFreqX = shapeX[0]
	numFrames = shapeX[1]

	shapeWD = np.shape(WD)
	numFreqD = shapeWD[0]
	rd = shapeWD[1]
	
	#initialization
	WD_update = 0
	HD_update = 0
	WH_update = 0
	HH_update = 0
	
	if len(WH) != 0:
		shapeWH = np.shape(WH)
		numFreqH = shapeWH[0]
		rh = shapeWH[1]
	else:
		WH = np.random.rand(numFreqD, rh)
		numFreqH = np.shape(WH)[0]
		WH_update = 1

	if numFreqD != numFreqX:
		print(numFreqD,'\n', numFreqX)
		print("ERROR: Dimensionality of the WD does not match X")
		exit(-1)
	elif numFreqH != numFreqX:
		print(numFreqH,'\n',  numFreqX)
		print("ERROR: Demensionality of the WH does not match X")
		exit(-1)

	if len(HD) != 0:
		WD_update = 1
	else:
		HD = np.random.rand(rd, numFrames)
		HD_update = 1

	if len(HH) == 0:
		HH = np.random.rand(rh, numFrames)
		HH_update = 1

	alpha = (rh + rd) / rd
	beta = rh / (rh + rd)

	#normalize W / H matrix

	for i in range(rd):
		WD[:, i] = np.divide(WD[:, i], np.linalg.norm(WD[:, i], 1))

	for i in range(rh):
		WH[:, i] = np.divide(WH[:, i], np.linalg.norm(WH[:, i], 1))

	count = 0
	rep = np.ones((numFreqX, numFrames))
	
	#start iteration
	while count < 300:
		#print(np.shape(WD), '\n', np.shape(HD), '\n', np.shape(WH), np.shape(HH))
		approx = (alpha * np.dot(WD, HD)) + (beta * np.dot(WH, HH))

		#update
		if WD_update:
			tmp1 = np.divide(X, approx) * np.transpose(alpha * HD)
			tmp2 = rep * np.transpose(alpha * HD)
			WD = np.multiply(WD, np.divide(tmp1, tmp2))
			
		if HD_update:
			#HD = HD .* ((alpha * WD)'* (X./approx))./((alpha * WD)'*rep + sparsity);
			tmp1 = np.dot(np.transpose(alpha * WD), np.divide(X, approx))
			tmp2 = np.dot(np.transpose(alpha * WD), rep) + sparsity
			HD = np.multiply(HD, np.divide(tmp1, tmp2))

		if WH_update:
			tmp1 = np.dot(np.divide(X, approx), np.transpose(beta * HH))
			tmp2 = np.dot(rep, np.transpose(beta * HH))
			WH = np.multiply(WH, np.divide(tmp1, tmp2))

		if HH_update:
			tmp1 = np.dot(np.transpose(beta * WH), np.divide(X, approx))
			tmp2 = np.dot(np.transpose(beta * WH), rep)
			HH = np.multiply(HH, np.divide(tmp1, tmp2))


		#normalize W matrix
		for i in range(rh):
			WH[:, i] = np.divide(WH[:, i], np.linalg.norm(WH[:, i], 1)) 
		for i in range(rd):
			WH[:, i] = np.divide(WD[:, i], np.linalg.norm(WD[:, i], 1))


		#normalize H matrix
		#calculate variation between iterations
		count = count + 1

		return WD, HD, WH, HH
