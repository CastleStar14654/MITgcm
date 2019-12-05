'''gendata.py, generating binary data in python
Author: Lin, XuChen
for magma ocean project
=========IMPORTANT========
ForTran and Python are different when dealing with array index
for ForTran, indices are column-major and start from 1,
for Python (C), indices are column-major and start from 0。

be careful with Kelvin & Celsius
'''

import numpy as np # array manipulation
from scipy.constants import * # scientific constants
import struct # to output binary data
from scipy.special import lpn # Legendre
from scipy.ndimage import gaussian_filter

BYTE_ORDER = '>' # big-endian, which is compatible with MITgcm
FORMAT = 'f' # float, i.e. 'real*4'
KELVIN = 273

def write(array, file_name, byte_order=BYTE_ORDER, format=FORMAT):
    flat_array = array.flatten()
    file = open(file_name, 'wb')
    for value in flat_array:
        byte = struct.pack(byte_order+format, value)
        file.write(byte)
    file.close()

delR = (10, 10, 10, 20, 20,
        30, 30, 40, 40, 50,
        50, 60, 60, 70, 70,
        80, 80, 90, 100, 100)

Ho = np.sum(delR);  #depth of ocean
nx=85;  # gridpints in x
ny=84;  # gridpints in y
xo=-85; yo=-84;  # degree, origin in x,y for ocean
dx=2;dy=2;  # degree, grid spacing in x, y
s0 = 2.1e6 # W/m^2, = sigma*T_max**2, T_max = 2500K

# =============================================================
# Flat bottom at z=-Ho
height = -Ho*np.ones((ny, nx))
# create a border ring of walls around edge of domain
height[0,:] = 0; height[-1,:] = 0
height[:,0] = 0; height[:,-1] = 0
write(height, 'bathy.bin')

# =============================================================
# Surface temperature relaxation
# ------ default temperature t_0 = T_0-KELVIN (s0*cos(x)*cos(y) = sigma*T_0**4), x & y in degree ----------
xs = np.arange(xo+dx/2, xo+dx/2 + dx*nx, dx).reshape((1,-1))
ys = np.arange(yo+dy/2, yo+dy/2 + dy*ny, dy).reshape((-1,1))
T_0 = (s0*np.cos(xs*pi/180)*np.cos(ys*pi/180)/sigma)**.25
write(T_0 - KELVIN, 'SST_relax.bin')
# ------ reciprocal of surface relaxation lambda_t = 4*sigma*T_0**3 ------
lambda_t =  4*sigma*T_0**3
write(lambda_t, 'SST_lambda.bin')

# =============================================================
'''viscosity (according to Liebske et al. (2005))
log_10{ eta (Pa*s) } = A + (B + C_1*P + C_2*P**2 + C_3*P**3)/(T-T_0), P in GPa,
    where A = -4.3, B = 3689 (27) K^-1, T_0 = 763 (3) K {or 490 Celsius},
        C_1 = 42 (44) K/GPa, C_2 = 28 (12) (K/GPa)^2, C_3 = -2.3 (7) (K/GPa)^3
Since our maximum pressure is about 0.8 GPa, we have
    B' = 3689 K^-1, C'_1 = 44 K/(GPa)

We assume A_molecure = eta/rho, let eddy viscosity A_r = A_molecure * 100, A_h = A_r*1e8
'''
# -------------- A_molecure -----------------
hr_ratio = 1e8 # A_h/A_r
a = 1e7 # m, planet radius
g = 24 # m/s^2
rho = 2670 # kg/m^3
para = {'A': -4.3, 'B': 3689, 'T': 763, 'C': 44} # temperature in K, pressure in GPa
# !!!!!!! Z, Y, X, L NOW !!!!!!!!
zs = []
for i in range(len(delR)):
    z = sum(delR[:i]) + delR[i]/2
    zs.append(z)
zs = np.array(zs).reshape((-1,1,1))
ys = ys.reshape((1,-1,1))
xs = xs.reshape((1,1,-1))

pressure = rho*g*zs/1e9 # GPa

coef = np.load('legendre_coef').reshape((1,1,1,-1)) # temperature coefficient, in K
ls = np.arange(coef.size).reshape((1,1,1,-1))
# temperature = SUM_l coef[0,0,0,l]*exp( -(l*(l+1))**.5 * hr_ratio**.5 *z/a)*P_l(cos(x)*cos(y))
# x, y in degree
P_ls = np.zeros((1,ny,nx,coef.size))
for numy in range(ny):
    for numx in range(nx):
        P_ls[0,numy,numx] = lpn(coef.size-1,
                                np.cos(ys[0,numy,0]*pi/180)
                                    *np.cos(xs[0,0,numx]*pi/180)
                                )[0]
temprature = np.sum(coef*np.exp(-(ls*(ls+1))**.5 * hr_ratio**.5 *zs.reshape((-1,1,1,1))/a)*P_ls,
                    axis=3)
temprature = gaussian_filter(temprature, 1, mode='nearest')
write(temprature - 41.5*pressure - KELVIN, 'hydrogTheta.bin')


eta = 10**(para['A']
           + (para['B'] + para['C']*pressure) / (temprature - para['T'])
           )
A_molecure = eta/rho
# -------------- A_r, REMEMBER TO COPY DATA IN 'ArNr' TO 'data' -----------------
A_r = np.average(A_molecure[:,10:-11,10:-11]*100, axis=(1,2))
file = open('ArNr', 'w')
for value in A_r:
    print(value, file=file)
file.close()
# -------------- A_h -----------------
A_h = A_molecure*100*hr_ratio
A_h[A_h > 1e10] = 1e10
write(A_h, 'viscAhfile.bin')
