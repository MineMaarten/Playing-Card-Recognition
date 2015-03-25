# bhutils
# Uitilties speciaal voor het practicum Beeldherkenning

import numpy as np
import cv2
import scipy.fftpack
from scipy import signal
import matplotlib.pyplot as plt

#-------------------------------------------------------
# cst: contrast strech an image
# Arguments:
# - <image> : image to be stretched
# to do list:
# test for multicolor
# support percentile based stretching

def cst(image):
	(h, w) = image.shape[:2]
	minval = np.min(image)
	maxval = np.max(image)
	contrast = maxval - minval
	stretched = image + 0
	for j in range(0, w-1):
	    for i in range (0, h-1):
	        stretched[i,j] = (image[i,j]-minval)*255./contrast
	return stretched

#-------------------------------------------------------
# translate: translate an image
# Arguments:
# - <image>  : image to be translated
# - <x>      : number of shifts in x-direction
# - <Y>      : number of shifts in y-direction

def translate(image, x, y):
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
	return shifted

#-------------------------------------------------------
# rotate: rotate an image
# Arguments:
# - <image>  : image to be rotated
# - <angle>  : rotation angle in degrees
# - <center> : [optional] center of rotation [default: center of image]
# - <scale>  : [optional] scale factor [default: 1.0, no additional scaling]

def rotate(image, angle, center = None, scale = 1.0):
	(h, w) = image.shape[:2]
	if center is None:
		center = (w / 2, h / 2)
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))
	return rotated

#-------------------------------------------------------
# resize: rezize an image
# Arguments:
# - <image>  : image to be resized
# - <width>  : [optional] width of the result image [default: unspecified]
# - <height> : [optional] height of the result image [default: unspecified]
# - <inter>  : [optional] interpolation method [default: cv2.INTER_AREA]
# If both <width> and <height> are not specified: do nothing
# If only <height> is specified: scale to specified height
# If only <width> is specified: scale to specified width
# If both <width> and <height> are specified: specified width is dominant

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation = inter)
	return resized

#-------------------------------------------------------
# spec1D: display spectrum of 1D signal
# Arguments:
# - <line>  : image to be resized
# - <side>  : [optional] single or double sided [default: single]
# - <window>: [optional] window to be applied before fft [default none]
# Window options:
# - none
# - Hanning

def spec1D(line, side = 'single', window = ' '):
	N = line.shape[0]

	if side != 'single':
		NS = N
		Fmax = 1.0
	else:
		NS = N/2
		Fmax = 0.5

	if window == 'Hanning':
		x1 = np.linspace(0.0, 1.0, N)
		line = line * 0.5 * (1. - np.cos(2.0*np.pi*x1))

	xf = np.linspace(0.0, Fmax, NS)
	yf = scipy.fftpack.fft(line)


# Plot original signal
	if window == 'Hanning':
		Header = 'Figure1, FFT with Hanning Window'
	else:
		Header = 'Figure1, FFT without Window'

	plt.figure(Header)
	plt.subplot(1,2,1)
	plt.title("Generated signal")
	plt.xlabel("Time")
	plt.ylabel("Amplitude")
	plt.plot(line, "g-")

# Plot magnitude spectrum
	plt.subplot(1,2,2)
	plt.title("Frequency spectrum")
	plt.xlabel("Frequency")
	plt.ylabel("Magnitude")
	plt.plot(xf, 2.0 * np.abs(yf[0:NS]))
	plt.show()
	done = True
	return done
	


