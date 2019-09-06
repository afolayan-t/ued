import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patch

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy.misc import face

import glob
import csv
import re

import sys 
import os
import copy
import time
import h5py
import skimage.feature
import skimage.filters
import skimage.measure
import socket  
import itertools
import math
from matplotlib.patches import Polygon
from IPython.display import clear_output
from PIL import Image, ImageOps

C = 299792458 #m/s

def h(t, t0 = 0):
    """returns 0 until time t passes t0 then it returns 1"""
    return [item>=t0 for item in t]

def exp_fit(x, t0, a, b, t1, t2):
    """linear combination of exponentials to fit function points"""
    x = np.array(x)
    y = np.reshape(h(x, t0)*np.array([a*np.exp(-x/t1)+b*np.exp(-x/t2)]), len(x))
    return y

def float_to_int(num):
    """Changes floats to ints, even when float is in a single value tuple."""
    num = str(num)
    num = num.split('.')
    return int(num[0])

def mm_to_ps(mm, zero=0, direction=-1):
    """changes mm to ps"""
    ps = direction * (np.array(mm)-np.array(zero))*2 / C * 1e9
    return ps

def ps_to_mm(ps, zero=0, direction=-1):
    """changes ps to mm"""
    mm = direction * (ps * C * 1e-9) / 2 + zero
    return mm

def gaus(x,a,x0,sigma,y0=0,k=0):
    """Gaussian Function as a python function"""
    x= np.array(x)
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + y0 + k*x

def checks_data_sizes(fnames, fnames_I0, delays):
    """
    Checks the shape of the fnames, fnames_I0 and delays arrays 
    to make sure each data shape matches data file sizes. If not,
    will return a data shape that works and a warning describing 
    the issue.
    """
    x = len(fnames)
    y = len(fnames_I0)
    z = len(delays)
    delays_n = delays
    if z != y or z != x:
        # if fnames_I0 is less than or equal to fnames
        if y < x:
            print("WARNING: Mismatched Data shape: fnames_I0 is less than fnames; reducing fnames and delays accordingly.")
            delays_n = []
            fnames_n = []
            for i in range(y):
                delays_n += [np.float64(fnames_I0[i].split('\\')[-1].split('_')[-2])]
            for j in range(len(delays_n)):
                fnames_n += [fnames[j]]
            return delays_n, fnames_n, fnames_I0
        # if fnames is less than or equal to fnames_I0
        elif x < y:
            print("WARNING: Mismatched Data shape: fnames is less than fnames_I0; reducing fnames_I0 and delays accordingly.")
            delays_n = []
            fnames_I0_n = []
            for i in range(x):
                delays_n += [np.float64(fnames[i].split('\\')[-1].split('_')[1].split('-')[-1])]
            for j in range(len(delays_n)):
                fnames_I0_n += [fnames_I0[j]]
            return delays_n, fnames, fnames_I0_n
    print("All Data shapes match")
    return delays, fnames, fnames_I0

#image loading and processing

def find_coords(image, coord, roisize=50, returnall=False, showerrors = True):
    """
        Takes in an image, coordinates, and returns either coordinates 
        or Coordinates, Amplitude, and Sigma, depending on if returnall
        is true or false. This function fits peaks with gaussians, if 
        unable to fit a peak, the function returns an error message, 
        and sets the peak to zero. 
    """
    #fitting the peak
    coord = list(np.array(coord,dtype=int))
    subimg = image.T[coord[0]-roisize/2:coord[0]+roisize/2,coord[1]-roisize/2:coord[1]+roisize/2]
    
    try:
        xproj = np.sum(subimg,axis=1)
        xpxl = range(coord[0]-roisize/2,coord[0]+roisize/2)

        mean = xpxl[np.argmax(xproj)] #sp.ndimage.measurements.center_of_mass(xproj) +np.min(xpxl)#sum(xpxl)/len(xpxl)
        sigma = 15
        y0 = min(xproj)
        a = max(xproj) - y0
        k=0
        poptx,pcovx = curve_fit(gaus,xpxl,xproj,p0=[a,mean,sigma,y0,k])

        yproj = np.sum(subimg,axis=0)
        ypxl = range(coord[1]-roisize/2,coord[1]+roisize/2)
        
        mean = ypxl[np.argmax(yproj)]#sp.ndimage.measurements.center_of_mass(yproj) +np.min(ypxl)
        #mean = sum(ypxl)/len(ypxl)
        sigma = 15
        y0 = min(yproj)
        a = max(yproj) - y0
        k=0
        popty,pcovy = curve_fit(gaus,ypxl,yproj,p0=[a,mean,sigma,y0,k])

        amplitude = np.mean([poptx[0], popty[0]])
        sigma = np.mean([poptx[2], popty[2]])
        coords = np.array([poptx[1],popty[1]])
    except:
        if showerrors == True:
            print('Fitting Failed!!!! for coords {};{}'.format(coord[0],coord[1]))
        amplitude = 0
        sigma = 0
        coords = np.array([0,0])
    if returnall == True:
        return coords, amplitude, sigma
    else:
        return coords
    
def fit_image(fname, fname_I0, indexes, roicoord,  roinames, roisize=60, correct_t0 = False):
    """
        This function takes in fname, fname_I0, indexes, roicoord, roinames, and returns a 
        data dictionary comprised of i0_pos, i0_sigma, i0_centroid, i0, i0_sum, delaycorrection,
        delay, center, amplitudes, sigma, length, coordinates, pixelsum.
    """
    #fitting the peaks from {indexes}
    data = dict()
    image = load_img(fname)
    image = np.array(image, dtype=float)
    
    #bg subtraction
    image -= np.mean([image[0:64,0:64],image[-64:,0:64],image[0:64,-64:],image[-64:,-64:]])
    
    #I normalization 
    i0 = load_img(fname_I0)

    pos_fitted,amplitude,sigma  = find_coords(i0,[280,226],roisize=200, returnall=True, showerrors=False)
    centroid = sp.ndimage.measurements.center_of_mass(i0.T[249:325,179:262])
    data['i0_pos'] = pos_fitted 
    data['i0_sigma'] = sigma
    data['i0_centroid'] = centroid
    data['i0'] = amplitude * sigma
    data['i0_sum'] = np.mean(i0.T[249:325,179:262]) - np.mean(i0.T[0:200,0:100])
    data['delaycorrection'] = (57.5-pos_fitted[1]) *0.0093
    data['delay'] = float(fname.split('\\')[-1].split('-')[-1].split('_')[-2])
    try: delays2 = np.array([float(fname.split('\\')[-1].split('-')[-3].split('_')[0]) for fname in fnames])
    except: delays2 = [-np.inf]
    
    totalI = np.sum(image)
    image /= totalI
    data['totalI'] = np.sum(image)

    #fitting first Bragg peaks
    roicoord_new = [find_coords(image, coord, roisize=50, returnall=False, showerrors = True) for coord in roicoord]
        
    #center recentered   
    center = np.mean(roicoord_new, axis=0)
    try: a = (roicoord_new[roinames.index('a')] - center) / 2
    except: a = np.array([0,0])
    try: b = (roicoord_new[roinames.index('b')] - center) / 2
    except: b = np.array([0,0])
    try: c = (roicoord_new[roinames.index('c')] - center) / 2
    except: c = np.array([0,0])

    data['center'] = center
    #now calculating positions for all other peaks and then fitting them
    data['amplitudes'] = []
    data['sigma'] = []
    data['length'] = []
    data['coordinates'] = []
    data['pixelsum'] = []
    count = 0
    for num, index in enumerate(indexes):
        pos = index[0] * a + index[1] * b + index[2] * c + center
        pos_fitted,amplitude,sigma = find_coords(image,pos, roisize=roisize, returnall=True)
        pos_fitted = [float_to_int(pos_fitted[0]),float_to_int(pos_fitted[1])]
        #print('pos_fitted:', pos_fitted[0], pos_fitted[1])
        pixelsum = np.mean(image.T[pos_fitted[0]-roisize/2:pos_fitted[0]+roisize/2,pos_fitted[1]-roisize/2:pos_fitted[1]+roisize/2])
        data['amplitudes'].append(amplitude)
        data['sigma'].append(sigma)
        data['length'].append(np.linalg.norm(pos_fitted - center))
        data['coordinates'].append(pos_fitted)
        data['pixelsum'].append(pixelsum)
    data['center_all'] = np.mean(data['coordinates'],axis=0)
    return data

def closest_point(points, x0,y0,x1,y1):
    """Finds closest point on the line for a selection of points"""
    line_direction = np.array([x1 - x0, y1 - y0], dtype=float)
    line_length = np.linalg.norm(line_direction)
    line_direction /= line_length
    
    n_bins = int(np.ceil(line_length))

    # project points on line
    #projections = np.array([(p[0] * line_direction[0] + p[1] * line_direction[1]) for p in points])
    projections = np.array([(p[0] * line_direction[0] + p[1] * line_direction[1]) for p in points])

    # normalize projections so that they can be directly used as indices
    projections -= np.min(projections)
    projections *= (n_bins - 1) / np.max(projections)
    return np.floor(projections).astype(int), n_bins

def rect_profile(x0, y0, x1, y1, width):
    """
        Takes in four points and a width and returns a rectangular 
        polygon, as well as 8 y and x points.
    """
    xd = x1 - x0
    yd = y1 - y0
    length = np.sqrt(xd**2 + yd**2)
    y00 = y0 + xd * width / length
    x00 = x0 - yd * width / length
    y01 = y0 - xd * width / length
    x01 = x0 + yd * width / length
    y10 = y1 - xd * width / length
    x10 = x1 + yd * width / length
    y11 = y1 + xd * width / length
    x11 = x1 - yd * width / length
    poly_points = [x00, x01, x10, x11], [y00, y01, y10, y11]
    poly = Polygon(((y00, x00), (y01, x01), (y10, x10), (y11, x11)))
    return poly, poly_points

def averaged_profile(image, x0, y0, x1, y1, width):
    """
        Takes in an image, 4 points and a width, and returns an
        averaged set of data,perpendicular to the profile created
        by the rect_profile function.
    """
    num = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    coords = list(zip(x, y))

    # Get all points that are in Rectangle
    poly, poly_points = rect_profile(x0, y0, x1, y1, width)
    points_in_poly = []
    for point in itertools.product(range(image.shape[0]), range(image.shape[1])):
        if poly.get_path().contains_point(point, radius=1) == True:
            points_in_poly.append((point[1], point[0]))

    # Finds closest point on line for each point in poly
    neighbours, n_bins = closest_point(points_in_poly, x0, y0, x1, y1)

    # Add all phase values corresponding to closest point on line
    data = [[] for _ in range(n_bins)]
    for idx in enumerate(points_in_poly):
        index = neighbours[idx[0]]
        data[index].append(image[idx[1][1], idx[1][0]])

    # Average data perpendicular to profile
    for i in enumerate(data):
        data[i[0]] = np.nanmean(data[i[0]])
    '''
    # Plot
    fig, axes = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)

    axes[0].imshow(image)
    axes[0].plot([poly_points[0][0], poly_points[0][1]], [poly_points[1][0], poly_points[1][1]], 'yellow')
    axes[0].plot([poly_points[0][1], poly_points[0][2]], [poly_points[1][1], poly_points[1][2]], 'yellow')
    axes[0].plot([poly_points[0][2], poly_points[0][3]], [poly_points[1][2], poly_points[1][3]], 'yellow')
    axes[0].plot([poly_points[0][3], poly_points[0][0]], [poly_points[1][3], poly_points[1][0]], 'yellow')
    axes[0].axis('image')
    axes[1].plot(data)'''
    return data

def calc_diffuse(data):
    """Takes in a set of data, and returns Bragg Peak Profiles for the diffuse scattering between peaks."""
    diffuse_profiles = [[] for _ in data['diffuse_sets']]
    
    braggcoords = data['braggcoords']
    braggnames = data['braggnames']
    for profnum, difuse_profile in enumerate(data['diffuse_sets']):
        tmp_prof = []
        for linenum, line in enumerate(difuse_profile):
                
            [x0, y0] = braggcoords[braggnames==line[0]][0]
            [x1, y1] = braggcoords[braggnames==line[1]][0]
            
            print('Image {}. profile {}'.format(data['indx'], linenum))

            #profile = averaged_profile(data['image'], x0, y0, x1, y1, data['halfwidth'])
            profile = skimage.measure.profile_line(data['image'].T,(x0,y0),(x1,y1),linewidth = data['halfwidth'])
            
            #multiply by scalling factor to account for the need of the profile width proportional to length
            #profile = np.dot(profile, 1/sp.linalg.norm((x0-x1,y0-y1)))
            #print(profile)
            x = np.linspace(-1, 1, len(profile))
            profile_interp = sp.interpolate.interp1d(x, profile, bounds_error=False)
            x_interpolated = np.linspace(-1, 1, data['npoints'])
            profile = np.array([profile_interp(x) for x in x_interpolated])
            
            tmp_prof.append(profile)
        diffuse_profiles[profnum] = np.mean(tmp_prof, axis=0)
    return diffuse_profiles

def generate_fnames (datafolder, delay2=-np.inf, minscannum=1, maxscannum=np.inf, ftype='tif'):
    """Takes data folders as strings, and returns lists of filenames fnames and fnames_I0"""
    try: i = np.shape(datafolder)[0]
    except: datafolder = [datafolder]
    fnames = []
    fnames_I0 = []
    for folder in datafolder:
        totalscans = len(glob.glob('{}/scan*/'.format(folder)))
        for scannum in range(max(minscannum,1),min(maxscannum,totalscans)+1):
            if delay2 < -1E10:
                fnames.extend(sorted(glob.glob('{}/scan{:03d}/images-ANDOR1/*ANDOR1_*.{}'.format(folder, scannum,ftype))))
                fnames_I0.extend(sorted(glob.glob('{}/scan{:03d}/I0/*ANDOR2_*.{}'.format(folder, scannum,ftype))))
            else:
                fnames.extend(sorted(glob.glob('{}/scan{:03d}/images-ANDOR1/*ANDOR1_longDelay-*-{:.8f}_2nd_pulse_delay-*.{}'.format(folder, scannum,delay2,ftype))))
                fnames_I0.extend(sorted(glob.glob('{}/scan{:03d}/I0/*ANDOR2_longDelay-*-0{:.4f}_*.{}'.format(folder, scannum,delay2,ftype))))
        #print(len(fnames))
    
    
    return fnames, fnames_I0

def load_img(fname):
    """takes in a fname and returns the image that corresponds to that fname."""
    if fname.split('.')[-1] == 'npy': return np.load(fname)
    else: return sp.ndimage.imread(fname)

#d-w analysis

def unique_values(name_list):
    """
        Takes in list of values, returns list of all unique values.
    """
    uniq_vals = []
    for i in range(len(name_list)):
        x = name_list[i]
        if x not in uniq_vals:
            uniq_vals += [x]
    return uniq_vals

def grouped_indices(name_list):
    """
        Takes in list of values, and returns a 2-D list 
        with the indexes of identical values grouped into separate lists.
    """
    grouped_ind = []
    values = unique_values(name_list)
    for i in range(len(values)):
        y = []
        for j in range(len(name_list)):
            if values[i] == name_list[j]:
                y += [j]
        grouped_ind += [y]
    return grouped_ind

def averaged_amplitudes(name_list, amplitudes):
    """
        Takes in a list of values and a list of amplitudes that have been
        grouped by peak and returns a nested list where the first index of
        each is the grouping label, and the second index is the averaged amplitudes.
    """
    indices = grouped_indices(name_list)
    averaged_amplitudes = []
    labels = []
    peak_amplitudes = []
    #organizing amplitude data points by delay orders
    for i in range(len(name_list)):
        x = []
        for j in range(len(amplitudes)):
            x += [amplitudes[j][i]]
        peak_amplitudes += [x]
        
    #grouping full amplitude data sets by momentum transfer
    for group in indices:
        amps = []
        for i in range(len(group)):
            amps += [peak_amplitudes[group[i]]]
        amps = np.true_divide(np.sum(amps, axis = 0), len(group))
        averaged_amplitudes += [amps]
        labels += [name_list[group[i]]]
    return averaged_amplitudes, labels

def g_2(pt, amplitudes, name_list, a, indexes):
    """
        Takes in a list of values and a list of amplitudes,
        a time range and a coefficient b and returns 
        a slice of averaged of amplitudes and a list of
        corresponding g2 values for plotting
    """
    #defining physical scale value for amplitudes
    ind = indexes[1]
    ang_dist = ind[0]**2+ind[1]**2+ind[2]**2
    label = int(name_list[1])
    scale_factor = ((ang_dist/a**2)/(label**2))
    b = (8*math.pi/3)*scale_factor
    pt = unique_values(pt)
    #initializing variables
    g2 = []
    new_amps = []
    #performing naturual log on amplitudes
    for i in range(len(amplitudes)):
        new_amp = []
        for j in range(len(amplitudes[i])):
            new_amp += [-math.log(amplitudes[i][j])]
        new_amps += [new_amp]
    g2_amp = []
    g2_amp_all = []
    #separating single points along certain time interval
    for i in range(len(amplitudes[1])):
        g2_amp = []
        for j in range(len(new_amps)):
            g2_amp += [new_amps[j][i]]
        g2_amp_all += [g2_amp]
    #creating q^2 points
    pt = (unique_values(pt))
    for k in range(len(amplitudes)):
        g2 += [(b*(pt[k]**2))]
    return g2_amp_all, g2


#ued data class
class Data:
    """Class for electron diffraction data analysis, represents and analyzes a set of data files."""
    
    #Defined Attributes
    minscannum = 1
    maxscannum = np.inf
    
    imcontrast = 10
    T0 = 0
    
    roicoord = [[640,520],
               [490,670],
               [352,520],
               [490,370]]
    roicoord = np.array(roicoord)
    
    #directions of first two rois
    roinames = ['a', 'b']
    rcolors = ['r','g','y','m','r','g','y','m']
        
    #Initializer / Instance Attributes
    def __init__(self, data_path, zero, zero2, a = 1, roisize = 60, ftype = 'tif', imcontrast = imcontrast, maxorder = [4,4,0], roicoord = roicoord, roinames = roinames, rcolors = rcolors):
        
        self.data_path = data_path
        self.ftype = ftype
        self.data = dict()
        
        self.a = a
        self.zero = zero
        self.zero2 = zero2
        self.T0 = 0
        self.imcontrast = imcontrast
        
        self.maxorder = maxorder
        self.roisize = roisize
        self.roicoord = roicoord
        self.roinames = roinames
        self.rcolors = rcolors
        
        #print(roisize, roicoord, roinames, rcolors, maxorder)
        
        self.minscannum = 1
        self.maxscannum = np.inf
        
        #filling self.images and self.data_libs
        try:runnum = data_path.split('\\')[0]
        except:runnum = '{}-{}'.format(data_path[0].split('\\')[-2],data_path[-1].split('\\')[-2])
        self.runnum = runnum
        
        self.fnames, self.fnames_I0 = generate_fnames(datafolder = data_path, ftype = self.ftype)
        
        self.delays = np.array([float(fname.split('\\')[-1].split('-')[-1].split('_')[-2]) for fname in self.fnames])
        self.delays_ps = mm_to_ps(self.delays,zero=zero)
        
        try: self.delays2 = np.array([float(fname.split('\\')[-1].split('-')[-3].split('_')[0]) for fname in self.fnames])
        except: self.delays2 = [-np.inf]
            
        image = np.mean([load_img(self.fnames[i]) for i in range(min(20, len(self.fnames)))], axis = 0)
        image = np.log(np.array(image, dtype = float) / np.mean(image))
        
        for indx, coord in enumerate(self.roicoord):
            rect = patch.Rectangle(coord - self.roisize*2/2, self.roisize*2, self.roisize*2, linewidth = 1, edgecolor = self.rcolors[indx], facecolor = 'none')
            self.roicoord[indx] = find_coords(image, coord, roisize =2*self.roisize, returnall = False, showerrors = False)
            rect = patch.Rectangle(self.roicoord[indx] - self.roisize/2, self.roisize, self.roisize, linewidth = 1, edgecolor = self.rcolors[indx], facecolor = 'none')
            
        #dictionary of images
        self.images = np.array([load_img(self.fnames[i]) for i in range(len(self.fnames))])
        center = np.mean(self.roicoord, axis = 0)
        self.center = center
        
        #defining reciprocal lattice
        try: a = (roicoord[roinames.index('a')] - center) / 2 
        except: a = np.array([0,0])
        try: b = (roicoord[roinames.index('b')] - center) / 2
        except: b = np.array([0,0])
        try: c = (roicoord[roinames.index('c')] - center) / 2
        except: c = np.array([0,0])
        
        #Lists for peak characterization
        bragg = []
        name = []
        indexes = []
        pt = []
        pt_hk = []
        
        #adding peak marks and labels to images
        maxorderi = maxorder[0]
        maxorderj = maxorder[1]
        maxorderk = maxorder[2]
        
        #recording preliminal peak labeling data
        for i in range(-maxorderi, maxorderi+1):
            for j in range(-maxorderj, maxorderj+1):
                for k in range(-maxorderk, maxorderk+1):
                    if (i==0 and j==0 and k==0) or (abs(i)+abs(j)+abs(k)) % 2 != 0:
                        continue
                    pos = (i*a + j*b + k*c) + center
                    q_hk = ((i*a)**2 + (j*b)**2 + (k*c)**2)**.5
                    q = ((q_hk[0])**2 + (q_hk[1])**2)**.5
                    pos_fitted = find_coords(image, pos, roisize = self.roisize)
                    #adding boxes and labels
                    if sp.linalg.norm(pos_fitted) == 0: pass
                    else:
                        bragg.append(pos)
                        name.append('{}'.format((float_to_int(q))))
                        indexes.append([i,j,k])
                        pt_hk.append(q_hk)
                        pt.append(q)
         
        self.bragg = np.array(bragg)
        self.name = np.array(name)
        self.indexes = np.array(indexes)
        self.pt = np.array(pt)
        self.pt_hk = np.array(pt_hk)
        
    def __repr__(self):
        """returns pertinent characteristics of the data object."""
        print("Runnum: {}".format(self.runnum))
        print("# of Images: {}".format(len(self.images)))
        print("Fnames: {}".format(len(self.fnames)))
        print("Fnames_I0: {}".format(len(self.fnames_I0)))
        print("Delays: {}".format(len(self.delays)))
        return '# of Peaks {}'.format(len(self.pt))
        
    def show_sample(self):
        """
        Displays a marked and labeled image from the collected data that is an average of 20 images from the image data.
        """
        image = np.mean([load_img(self.fnames[i]) for i in range(min(20, len(self.fnames)))], axis = 0)
        image = np.log(np.array(image, dtype = float) / np.mean(image))
        
        fig, ax = plt.subplots(1,1,figsize=(9.8, 8), tight_layout = True)
        vmax = np.max(image)/self.imcontrast
        ax.imshow(image, vmax = vmax)
        
        #defining reciprocal lattice
        try: a = (self.roicoord[self.roinames.index('a')] - self.center) / 2
        except: a = np.array([0,0])
        try: b = (self.roicoord[self.roinames.index('b')] - self.center) / 2
        except: b = np.array([0,0])
        try: c = (self.roicoord[self.roinames.index('c')] - self.center) / 2
        except: c = np.array([0,0])
            
        maxorderi = self.maxorder[0]
        maxorderj = self.maxorder[1]
        maxorderk = self.maxorder[2]
        
        for i in range(-maxorderi,maxorderi+1):
            for j in range(-maxorderj,maxorderj+1):
                for k in range(-maxorderk,maxorderk+1):
                    if (i==0 and j==0 and k==0) or (abs(i)+abs(j)+abs(k)) % 2 !=0:
                        continue
                    self.pos = (i*a + j*b + k*c) + self.center
                    self.q_hk = ((i*a)**2 + (j*b)**2 + (k*c)**2)**.5
                    self.q = ((self.q_hk[0])**2 + (self.q_hk[1])**2)**.5
                    self.pos_fitted = find_coords(image, self.pos, roisize = self.roisize)
                    #adding boxes and labels
                    rect = patch.Rectangle(self.pos - self.roisize/2, self.roisize, self.roisize, linewidth=1, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                    #ax.text(self.q_hk[0]+10,self.q_hk[1]+10, '{}'.format(float_to_int(self.q)), color='w', fontsize=7)
                    
                    
        fig.show()
        return fig
        
    def fit_data(self):
        """Fits data and returns data libraries for all data and all delays"""
        #labeling peaks
        #storing averaged data:
        alldata = dict()
        alldelays = dict()   
        #storing NOT averaged data:
        alldata_na = dict()
        alldelays_na = dict()
        
        alldata[self.runnum] = dict()
        alldelays[self.runnum] = dict()
        alldata_na[self.runnum] = dict()
        alldelays_na[self.runnum] = dict()
        
        for delay2 in np.unique(self.delays2):
            self.fnames, self.fnames_I0 = generate_fnames(datafolder = self.data_path, delay2 = delay2, ftype = self.ftype)
            self.delays = np.array([float(fname.split('//')[-1].split('-')[-1].split('_')[-2]) for fname in self.fnames])
            
            #guaranteeing data shapes
            self.delays, self.fnames, self.fnames_I0 = checks_data_sizes(fnames = self.fnames, fnames_I0 = self.fnames_I0, delays = self.delays)
            self.delays_ps = mm_to_ps(self.delays, zero = self.zero)
            
            self.delay2_ps = np.round(mm_to_ps(delay2, zero = self.zero2, direction = 1), 3)
            #for cluster implementation
            #pushdata = dict(delays = self.delays, fnames = self.fnames, fnames_I0 = self.fnames_I0, ftype = self.ftype, indexes = self.indexes, roicoord = self.roicoord, roinames = self.roinames, roisize = self.roisize)
            #tmp = dv.map(lambda delaynum: fit_image(self.fnames[delaynum], fnames_I0[delaynum], self.indexes, self.roicoord, self.roinames, roisize = self.roisize), range(len(self.delays)))
            
            tmp = map(lambda delaynum: fit_image(self.fnames[delaynum], self.fnames_I0[delaynum], self.indexes, self.roicoord, self.roinames, roisize = self.roisize), range(len(self.delays)))
            
            keylist = tmp[0].keys()
            for key in keylist:
                for dnum, item in enumerate(tmp):
                    if dnum == 0:
                        self.data[key] = []
                    self.data[key].append(tmp[dnum][key])
                    
            for key in self.data.keys():
                self.data[key] = np.array(self.data[key])
            
            print('Images before filtering: {}'.format(len(self.data['i0'])))
            
            #max deviations from avg values
            deviations = dict()
            deviations['i0'] = 0.8
            deviations['i0_centroid'] = 0.8
            deviations['center'] = 0.1
            
            avg_values = dict()
            filt = []
            for key, item in deviations.items():
                avg_values[key] = np.mean(self.data[key], axis = 0)
                tmp = np.logical_and(self.data[key] >= avg_values[key] * (1-deviations[key]), self.data[key] <= avg_values[key]*(1 + deviations[key]))
                try:
                    _=np.shape(tmp)[0]
                    tmp = [np.prod(tmp_element, axis = 0) for tmp_element in tmp]
                except: pass
                filt.append(tmp)
            filt = np.array(np.prod(filt, axis = 0), dtype = bool)
            
            #Removing all bad data:
            for key, item in self.data.items(): self.data[key] = np.array(item)[filt]
            self.delays = np.array(self.delays)[filt]
            self.delays_ps = np.array(self.delays_ps)[filt]
            select_neg_full = self.delays_ps < self.T0
            self.fnames = np.array(self.fnames)[filt]
            self.fnames_I0 = np.array(self.fnames_I0)[filt]
            
            print('Images after filtering: {}'.format(len(self.data['i0'])))
            
            #filtering by deviation from neg time
            
            deviations_rel = dict()
            deviations_rel['length'] = 0.1
            deviations_rel['sigma'] = 0.1
            
            neg_values = dict()
            filt = []
            for key, item in deviations_rel.items():
                neg_values[key] = np.mean(self.data[key][select_neg_full], axis = 0)
                tmp = np.logical_and(self.data[key] >= neg_values[key] * (1 - deviations_rel[key]), self.data[key] <= neg_values[key] * (1+ deviations_rel[key]))
                try:
                    _=np.shape(tmp)[0]
                    tmp = [np.prod(tmp_element, axis = 0) for tmp_element in tmp]
                except: pass
                filt.append(tmp)
            filt = np.array(np.prod(filt, axis = 0), dtype = bool)
            
            for key, item in self.data.items():
                self.data[key] = np.array(item)[filt]
            self.delays = np.array(self.delays)[filt]
            self.delays_ps = np.array(self.fnames)[filt]
            select_neg_full = self.delays_ps < self.T0
            self.fnames = np.array(self.fnames)[filt]
            self.fnames_I0 = np.array(self.fnames_I0)[filt]
            print('Images after filtering stage 2: {}'.format(len(self.data['i0'])))
            
            #average same delays
            data_avg = dict()
            delays_avg = np.unique(self.data['delay'])
            delays_ps_avg = mm_to_ps(delays_avg, zero = self.zero)
            
            for key,item in self.data.items():
                if key in []:
                    continue
                try:
                    i = np.shape(self.data[key])[2]
                    data_avg[key] = np.zeros((len(delays_ps_avg), len(self.indexes),i))
                except:
                    try:
                        i = np.shape(self.data[key])[1]
                        data_avg[key] = np.zeros((len(delays_ps_avg),i))
                    except:
                        data_avg[key] = np.zeros(len(delays_ps_avg))
                for delaynum, delay in enumerate(delays_avg):
                    data_avg[key][delaynum] = np.mean(self.data[key][self.data['delay'] == delay], axis = 0)
            
            #normalize to negative delays        
            data_avg_norm = copy.deepcopy(data_avg)
            for key, item in data_avg_norm.items():
                if key in ['i0','i0_pos','i0_centroid','center','coordinates','delay','totalI']:
                    continue
                
                select_neg = delays_ps_avg < self.T0
                if (select_neg.any() == True):
                    factor = np.mean(item[select_neg], axis = 0)
                else:
                    factor = n.mean(item, axis = 0)
                if factor.all > 0:
                    pass
                else:
                    factor = item[-1]
                for delaypoint in item:
                    delaypoint /= factor
                 
            alldelays[self.runnum][self.delay2_ps]  = copy.deepcopy(delays_ps_avg)
            alldata[self.runnum][self.delay2_ps]  = copy.deepcopy(data_avg_norm)
            alldelays_na[self.runnum][self.delay2_ps]  = copy.deepcopy(self.delays_ps)
            alldata_na[self.runnum][self.delay2_ps]  = copy.deepcopy(self.data)
            
            self.alldelays = alldelays
            self.alldata = alldata
            self.alldelays_na = alldelays_na
            self.alldata_na = alldelays_na
            
            print('Done.')
        return alldelays, alldata, alldelays_na, alldata_na
    
    def display_prelim_data(self):
        """Returning data plots for amplitudes, sigma, pixelsum, and vector length"""
        
        fig_i, ax_i = plt.subplots(2, 2, figsize = (9.8, 8), tight_layout = True)
        
        selectpeaks = self.name
        select_array = np.array([item in selectpeaks for item in self.name])
        
        for key, element in self.alldata[self.runnum].items():
            ax_i[0,0].plot(self.alldelays[self.runnum][key], element['amplitudes'][::,select_array])
            ax_i[0,1].plot(self.alldelays[self.runnum][key], element['sigma'][::,select_array])
            ax_i[1,0].plot(self.alldelays[self.runnum][key], element['pixelsum'][::,select_array])
            ax_i[1,1].plot(self.alldelays[self.runnum][key], element['length'][::,select_array])
            
        ax_i[0,0].set_title('Intensity')
        ax_i[0,0].set_ylabel('I/I0')
        
        ax_i[0,1].set_title('Sigma')
        ax_i[0,1].set_ylabel('S/S0')
        
        ax_i[1,0].set_title('Pixel Sum')
        ax_i[1,0].set_ylabel('I/I0')
        
        ax_i[1,1].set_title('Vector Length')
        ax_i[1,1].set_ylabel('L/L0')
        
        fig_i.suptitle('Run {}'.format(self.runnum))
        fig_i.show()
              
        
    def display_d_w(self, display_count = 5, showall = False, num_bins = 10, t0 = 0):
        """Returns Debye-Waller Analysis plots and statistics"""
        
        selectpeaks = self.name
        select_array = np.array([item in selectpeaks for item in self.name])
        
        for key, element in self.alldata[self.runnum].items():
            amp_avg, labels = averaged_amplitudes(self.name, element['amplitudes'][::, select_array])
            amps, g2 = g_2(self.pt, amp_avg, self.name, self.a, self.indexes)
            t_delays = np.array(self.alldelays[self.runnum][key])
            
        fig_j, ax_j = plt.subplots(1, 3, figsize = (9.8, 3), tight_layout = True)
        for key, element in self.alldata[self.runnum].items():
            for i in range(len(amp_avg)):
                ax_j[0].plot(self.alldelays[self.runnum][key], amp_avg[i])
                ax_j[0].legend(labels, loc = 1, prop = {'size': 6})
            ax_j[0].set_ylabel('I/I0')
            ax_j[0].set_xlabel('Delays [ps]')
            
            y = []
            for i in range(0, display_count):
                y += [round(self.alldelays[self.runnum][key][i], 4)]
            y = np.array(y)
            for i in range(0, display_count):
                x = np.array(amps[i])
                x[0::] += i*.1
                z = np.polyfit(g2, x, 1)
                poly_fit = np.array([z[1] + z[0]*xi for xi in g2])
                
                ax_j[1].scatter(g2, x)
                ax_j[1].legend(y, loc = 1, prop = {'size': 6})
            for i in range(0, display_count):
                x = np.array(amps[i])
                x[0::] += i*.1
                z = np.polyfit(g2, x, 1)
                poly_fit = np.array([z[1] + z[0]*xi for xi in g2])
                ax_j[1].plot(g2, poly_fit)
                
            ax_j[1].set_ylabel('-ln(I/I0)')
            ax_j[1].set_xlabel('G_2 [A^-2]')
            
            amplitude_slopes = [(np.polyfit(g2, i, 1))[0] for i in amps]
            covarr = [((np.polyfit(g2, i, 1, cov = True)[1][0][0])**.5) for i in amps]
            
            a = .015
            b = .015
            t1 = 15
            t2 = 25
            
            popt, pcov = curve_fit(exp_fit, t_delays, amplitude_slopes, p0 = np.array([t0, a, b, t1, t2]))
            
            x = np.linspace(-5, 100, 1051)
            t0, a, b, t1, t2 = popt
            nonlin_fit = exp_fit(x, t0, a, b, t1, t2)
            
            '''amplitude_slopes = [(np.polyfit(g2, i, 1))[0] for i in amps]
            covarr = [((np.polyfit(g2, i, 1, cov = True)[1][0][0])**.5) for i in amps]'''
            
            ax_j[2].errorbar(self.alldelays[self.runnum][key], amplitude_slopes, yerr = covarr, fmt = 'o')
            ax_j[2].plot(x, nonlin_fit)
            ax_j[2].set_ylabel('<u^2> [A^-2]')
            ax_j[2].set_xlabel('Delay [ps]')
            
            fig_j.suptitle('Run {}'.format(self.runnum))
            fig_j.show()
            
            if showall == True:
                fig_k, ax_k = plt.subplots(1, 2, figsize = (7.8, 3), tight_layout = True)
                for key, element in self.alldata[self.runnum].items():
                    #plotting averaged d-w linear curve
                    poly_fits = []
                    avgd_t = np.array(np.true_divide(np.sum(amps, axis = 0), len(amps)))
                    for i in range(len(amps)):
                        z = np.polyfit(g2, amps[i], 1)
                        poly_fits += [[z[1] + z[0]*xi for xi in g2]]
                        poly_fits1 = np.array(np.true_divide(np.sum(poly_fits, axis = 0), len(amps)))
                    
                    ax_k[0].scatter(g2, avgd_t)
                    ax_k[0].plot(g2, poly_fits1)
                    
                    ax_k[0].set_ylabel('-ln(I/I0)')
                    ax_k[0].set_xlabel('G_2 [A^-2]')
                    
                    ax_k[0].set_title('Average')
                    #generating r^2 gistogram
                    
                    hist = []
                    
                    for i in range(len(amps)):
                        z = np.polyfit(g2, amps[i], 1, full = True)
                        avg_z = np.polyfit(g2, np.mean(amps, axis = 0), 1, full = True)
                        hist += [1 - ((z[0][1]**2)/(avg_z[0][1]**2))]
                        
                    ax_k[1].hist(hist, num_bins, facecolor = 'blue', alpha = .5)
                    ax_k[1].set_xlabel('R^2')
                    ax_k[1].set_title('R Squared Histogram')
                    
                    fig_k.show()
            
        