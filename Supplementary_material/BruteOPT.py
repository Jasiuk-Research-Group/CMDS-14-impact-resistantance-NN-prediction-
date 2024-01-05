import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import matplotlib.pyplot as plt
# from mipego import ParallelBO, ContinuousSpace, OrdinalSpace, NominalSpace, RandomForest , AnnealingBO
import time
# from skimage.io import imread
# from skimage.measure import label
# from skimage.morphology import disk, skeletonize , dilation

# Regenerate perimeters?
regenerate = False


# Read saved stuffs
# Image dictionary
with open('ML_Model/ImgDict.pkl', 'rb') as f:
    img_dict = pickle.load(f)
nImages = len(list(img_dict.keys()))
All_keys = list( img_dict.keys() )

# GRU model
GRU = tf.keras.models.load_model( "ML_Model/GRU_model.h5" )
print('Loaded trained GRU model!')

# Finally, read all scalers
f = open('ML_Model/Scalers.npy','rb')
xScalers , encoded_scaler , yScalers = np.load( f , allow_pickle=True )
f.close()

if regenerate:
	# Image autoencoder
	autoencoder = tf.keras.models.load_model( "ML_Model/ImgEncoder" )
	print('Loaded saved image encoder!')

	# Encode all images
	print('Encoding all image inputs now...')
	All_img_raw = np.zeros([nImages,128,128])
	ii = 0
	inverse_map = {}
	for k , v in img_dict.items():
		inverse_map[ k ] = ii
		All_img_raw[ ii , : , : ] = v
		ii += 1
	All_img_encoded = autoencoder.encoder( All_img_raw )
	All_img_encoded = encoded_scaler.transform( All_img_encoded )


	# Get design perimeter
	# Fixerd parameters
	center = [ 232 , 363 ]
	N = 465
	dx = 20. / 444
	perimeter = np.zeros( nImages )
	for key in All_keys:
		img = imread('G:/Jasiuk Research Group/Simulation files/Static vs Dynamic response/GeometricDescriptor/CrossSections/K' + key + '.png')
		# Crop image
		img = img[ center[0]-N//2:center[0]+N//2 , center[1]-N//2:center[1]+N//2 ]

		# Binarize
		img = skeletonize( img < 1 )

		p = np.sum(img) * 0.04887349559722181 # Perimeter in mm
		perimeter[ inverse_map[ key ] ] = p
	np.save( 'ML_Model/Perimeter.npy' , perimeter )
	np.save( 'ML_Model/EncodedImg.npy' , All_img_encoded )
	np.save( 'ML_Model/InvMap.npy' , inverse_map )

else:
	perimeter = np.load('ML_Model/Perimeter.npy')
	All_img_encoded = np.load('ML_Model/EncodedImg.npy')
	inverse_map = np.load('ML_Model/InvMap.npy' , allow_pickle=True ).item()


#####################################################################################
# Material constants
E = 109778.e6 #Pa
rho = 4428. # kg/m^3
c0 = np.sqrt( E / rho ) # m/s
ey = 0.2 / 100.
L = 10.
t_elastic = ( L / 1000. ) / c0 # s

# Helper functions
# Create array inputs
def BuildArrayInputs( my_thickness , my_strain_rate , final_strain ):
	arr_inputs = np.zeros( [1 , 50 , 6] )

	arr_inputs[0,:,0] = my_thickness
	arr_inputs[0,:,1] = np.log10( my_strain_rate )
	arr_inputs[0,:,2] = final_strain
	arr_inputs[0,:,3] = np.linspace( 0. , final_strain / my_strain_rate , 50 )
	arr_inputs[0,:,4] = np.linspace( 0. , final_strain , 50 )
	arr_inputs[0, arr_inputs[0,:,3] >= t_elastic , 5 ] = 1
	return arr_inputs


# Get specific energy absorption
def SpecificAbsorption( key , my_thickness , my_strain_rate , final_strain ):
	key_id = inverse_map[key]
	img_encoded = All_img_encoded[ key_id ].reshape([1,100]) # Already scaled

	# Build other array inputs
	arrs = BuildArrayInputs( my_thickness , my_strain_rate , final_strain )
	# Scale
	for i in range(len(xScalers)):
		curr_scaler = xScalers[i]
		arrs[:,:,i] = curr_scaler.transform( arrs[:,:,i] )

	# Predicting
	prediction = GRU.predict( [img_encoded , arrs] )

	# Undo scaling
	for i in range(len(yScalers)):
		curr_scaler = yScalers[i]
		prediction[:,:,i] = curr_scaler.inverse_transform( prediction[:,:,i] )	

	# Obtain stored + absorbed energy at last increment
	totE = np.sum( np.mean( prediction[0,-5:,1:] , axis=0 ) ) # Index 0 is reaction force

	# Get volume
	vol = perimeter[ key_id ] * my_thickness

	if vol >= min_vol and vol <= max_vol:
		return totE / vol
	else:
		return np.nan


# Brute optimization
def Brute( final_strain , my_strain_rate , N_thickness , fn ):
	t_arry = np.linspace( 0.25 , 0.75 , N_thickness )

	repetition = 0
	SEA = np.zeros( [ N_thickness , nImages ] )
	key_int = np.zeros_like( SEA )
	key_int2 = np.zeros_like( SEA )
	thickness_val = np.zeros_like( SEA )

	# Write general inputs once
	img_input = All_img_encoded
	template = BuildArrayInputs( 1. , my_strain_rate , final_strain )
	arr_inputs = np.zeros( [ nImages , 50 , 6 ] )
	for i in range( nImages ):
		arr_inputs[ i , : , : ] = template

	# Scale arr input
	for i in range( 1 , len(xScalers)):
		curr_scaler = xScalers[i]
		arr_inputs[:,:,i] = curr_scaler.transform( arr_inputs[:,:,i] )
	thickness_scaler = xScalers[0]

	key_template2 = []
	for k in All_keys:
		key_template2.append(int(k))
	key_template = np.arange( nImages )

	all_t = []
	for my_thickness in t_arry:
		print('Thickness loop ' , repetition+1 , '/'  , N_thickness )
		start_time = time.time()

		# Set current thickness
		arr_inputs[ : , : , 0 ] = my_thickness

		# Scale thickness in arr input
		arr_inputs[:,:,0] = thickness_scaler.transform( arr_inputs[:,:,0] )

		# Predicting
		ti = time.time()
		prediction = GRU.predict( [img_input , arr_inputs] , batch_size = 100 )
		all_t.append( time.time() - ti )

		# Undo scaling
		for i in range(len(yScalers)):
			curr_scaler = yScalers[i]
			prediction[:,:,i] = curr_scaler.inverse_transform( prediction[:,:,i] )	

		# Obtain stored + absorbed energy at last increment
		totE = np.sum( np.mean( prediction[:,-5:,1:] , axis=1 ) , axis = -1 )

		# SEA
		vol = perimeter * my_thickness
		sea = totE / vol

		invalid_flag = np.logical_not( (vol >= min_vol) * (vol <= max_vol) )
		sea[ invalid_flag ] = np.nan

		# Store data
		SEA[ repetition , : ] = sea
		key_int[ repetition , : ] = key_template
		key_int2[ repetition , : ] = key_template2
		thickness_val[ repetition , : ] = my_thickness

		opt_time = time.time() - start_time
		print('Inner loop took ' , opt_time , 's\n')
		repetition += 1
	np.save( fn , [ SEA , key_int , key_int2 , thickness_val ] )

	tot_infer_time = np.sum(all_t)
	print( tot_infer_time )
	print( tot_infer_time / ( N_thickness * 660 ) )


#####################################################################################

# Begin optimization
# Specify strain rate and final strain
global max_vol , min_vol
my_strain_rate = 1.e5
final_strain = 0.15

# Define range
min_vol = 0.
max_vol = 1000.

# print( np.min( perimeter) * 0.25 )
# print( np.mean( perimeter) * 0.25 )
# print( np.mean( perimeter) * 0.75 )
# print( np.max( perimeter) * 0.75 )


Brute( final_strain , my_strain_rate , 50 , 'test' )



# N_t = 50
# t_arry = np.linspace( 0.25 , 0.75 , N_t )


# # # Sweep 1: fix final strain, vary strain rate
# # expo = np.linspace( 2. , 5. , 10 )
# # All_strain_rate = np.power( 10. , expo )
# # final_strain = 0.2

# # # Sweep 2: fix strain rate, vary final strain
# # All_final_strain = np.linspace( 0.05 , 0.2 , 10 )
# # my_strain_rate = 1e5

# # Sweep 3: fix strain rate and final strain, vary max volume
# lb , ub = [] , []
# for img in img_dict.values():
# 	si = np.sum( img )
# 	lb.append( si * t_arry[0] )
# 	ub.append( si * t_arry[-1] )

# All_max_vol = np.linspace( np.mean(lb) , np.mean(ub) , 10 )

