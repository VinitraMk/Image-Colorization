#LAB-RGB conversions
import torchvision
import torch

def preprocess_lab(lab):
		# L_chan: black and white with input range [0, 100]
		# a_chan/b_chan: color channels with input range ~[-110, 110], not exact
		# [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
		scale = torch.tensor([50,110,110]).reshape([1,1,1,3])
		offset = torch.tensor([-1,0,0]).reshape([1,1,1,3])
		return lab/scale + offset
		

def deprocess_lab(labp):
		scale = torch.tensor([50,110,110]).reshape([1,1,1,3])
		offset = torch.tensor([-1,0,0]).reshape([1,1,1,3])
		lab = (labp - offset)*scale
		return lab

def rgb_to_lab(srgb):

	srgb_pixels = torch.reshape(srgb, [-1, 3])

	linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor)
	exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor)
	rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
	
	rgb_to_xyz = torch.tensor([
				#    X        Y          Z
				[0.412453, 0.212671, 0.019334], # R
				[0.357580, 0.715160, 0.119193], # G
				[0.180423, 0.072169, 0.950227], # B
			]).type(torch.FloatTensor)
	
	xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)
	

	# XYZ to Lab
	xyz_normalized_pixels = torch.mul(xyz_pixels, torch.tensor([1/0.950456, 1.0, 1/1.088754]).type(torch.FloatTensor))

	epsilon = 6.0/29.0

	linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor)

	exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor)

	fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask
	# convert to lab
	fxfyfz_to_lab = torch.tensor([
		#  l       a       b
		[  0.0,  500.0,    0.0], # fx
		[116.0, -500.0,  200.0], # fy
		[  0.0,    0.0, -200.0], # fz
	]).type(torch.FloatTensor)
	lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor)
	#return tf.reshape(lab_pixels, tf.shape(srgb))
	return preprocess_lab(torch.reshape(lab_pixels, srgb.shape))

def lab_to_rgb(labp):
		lab = deprocess_lab(labp)
		lab_pixels = torch.reshape(lab, [-1, 3])
		# convert to fxfyfz
		lab_to_fxfyfz = torch.tensor([
			#   fx      fy        fz
			[1/116.0, 1/116.0,  1/116.0], # l
			[1/500.0,     0.0,      0.0], # a
			[    0.0,     0.0, -1/200.0], # b
		]).type(torch.FloatTensor)
		fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor), lab_to_fxfyfz)

		# convert to xyz
		epsilon = 6.0/29.0
		linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor)
		exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor)


		xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29.0)) * linear_mask + ((fxfyfz_pixels+0.000001) ** 3) * exponential_mask

		# denormalize for D65 white point
		xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor))


		xyz_to_rgb = torch.tensor([
			#     r           g          b
			[ 3.2404542, -0.9692660,  0.0556434], # x
			[-1.5371385,  1.8760108, -0.2040259], # y
			[-0.4985314,  0.0415560,  1.0572252], # z
		]).type(torch.FloatTensor)

		rgb_pixels =  torch.mm(xyz_pixels, xyz_to_rgb)
		# avoid a slightly negative number messing up the conversion
		#clip
		rgb_pixels[rgb_pixels > 1] = 1
		rgb_pixels[rgb_pixels < 0] = 0

		linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor)
		exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor)
		srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (((rgb_pixels+0.000001) ** (1/2.4) * 1.055) - 0.055) * exponential_mask
	
		srgb_pixels[srgb_pixels>1]=1
		srgb_pixels[srgb_pixels<0]=0

		return torch.reshape(srgb_pixels, lab.shape)

