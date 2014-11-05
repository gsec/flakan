#! /usr/bin/env python
# -*- coding: utf_8 -*-
"""Exponential 2D Fit"""

from scipy.optimize import leastsq
import numpy as np

class Exp2D():
	def fitfunc(self, param, x):
		x0, y0, amp, exponent = param
		return y0 + amp * np.exp( exponent * (x-x0) )

	def errfunc(self, p, x, y):
		return y - self.fitfunc(p, x)

	def fit(self, x, y = None, x0 = None, y0 = None, amp = None, exponent = None):
		if y == None:		x,y = x[:,0], x[:,1]

		if x0 == None:		x0 = x[np.argmax(y)]
		if amp == None:		amp = max(y)-min(y)
		if y0 == None:		y0 = max(y)-amp
		if exponent == None:
			if x[0] < x[-1]: #decay
				exponent = -100./x.max() #TODO: get some value based on data
			else:
				exponent = x.max()/100 #Does this work?
		guess = x0, y0, amp, exponent

		p=['x0','y0','amp','exponent']

		best, cov, info, errmsg, success = leastsq(self.errfunc, guess, args=(x, y), full_output=1)

		print
		print('exp2d-fit:')

		if success==1:
			print "Converged within %10i iterations"%(info['nfev'])
		else:
			print "Not converged"
			print errmsg

		chisq=sum(info["fvec"]*info["fvec"])

		dof=len(x)-len(p)
		# chisq, sqrt(chisq/dof) agrees with gnuplot
		print "Converged with chi squared ",chisq
		print "degrees of freedom, dof ", dof
		print "RMS of residuals (i.e. sqrt(chisq/dof)) ", np.sqrt(chisq/dof)
		print "Reduced chisq (i.e. variance of residuals) ", chisq/dof
		print

		err = []
		for i,pmin in enumerate(best):
			err.append(np.sqrt(cov[i,i])*np.sqrt(chisq/dof))
			print "%2i %-10s %12f +/- %10f"%(i,p[i],pmin,np.sqrt(cov[i,i])*np.sqrt(chisq/dof))

		return(p,best,err)


	def fitdata(self, x, best, err=None, interpolate=True):
		if interpolate==True:
			x_fit = np.arange(min(x), max(x))
		else:
			x_fit = x
		y_fit = self.fitfunc(best, x_fit)
		return x_fit, y_fit

class Exp2D_x0y0const(Exp2D):
	def fitfunc(self, param, x):
		amp, exponent = param
		return amp * np.exp( exponent * (x) )

	def fit(self, x, y = None, amp = None, exponent = None):
		if y == None:		x,y = data[:,0], data[:,1]
		if amp == None:		amp = max(y)-min(y)
		if exponent == None:
			if x[0] < x[-1]: #decay
				exponent = -100./x.max() #TODO: get some value based on data
			else:
				exponent = x.max()/100 #Does this work?
		guess = amp, exponent

		p=['amp','exponent']

		best, cov, info, errmsg, success = leastsq(self.errfunc, guess, args=(x, y), full_output=1)

		print
		print('exp2d-fit:')

		if success==1:
			print "Converged within %10i iterations"%(info['nfev'])
		else:
			print "Not converged"
			print errmsg

		chisq=sum(info["fvec"]*info["fvec"])

		dof=len(x)-len(p)
		# chisq, sqrt(chisq/dof) agrees with gnuplot
		print "Converged with chi squared ",chisq
		print "degrees of freedom, dof ", dof
		print "RMS of residuals (i.e. sqrt(chisq/dof)) ", np.sqrt(chisq/dof)
		print "Reduced chisq (i.e. variance of residuals) ", chisq/dof
		print

		err = []
		for i,pmin in enumerate(best):
			err.append(np.sqrt(cov[i,i])*np.sqrt(chisq/dof))
			print "%2i %-10s %12f +/- %10f"%(i,p[i],pmin,np.sqrt(cov[i,i])*np.sqrt(chisq/dof))

		return(p,best,err)




'''
limits={'x_min': best[0]-50, 'x_max': best[0]+50}
x_inrange = clip(data[:,0],limits['x_min'],limits['x_max'])


fig = plt.figure()
ax = fig.add_subplot(111)

# plot the spectra in the given range
#plot(x,y)
#plot(data[:,0], data[:,1])
ax.plot(x_inrange, data[:,1])
xlabel(u'Wellenlänge [nm]')
ylabel(u'Intensität [a.u.]')

x_fit = arange(limits['x_min'],limits['x_max'])
y_fit = fitfunc_gauss1d(best, x_fit)
ax.plot(x_fit,y_fit)


s = ''
for i,pmin in enumerate(best):
        s += "%-5s %.5f +/- %.5f\n"%(param[i],pmin,err[i])

fwhm = 2 * sqrt(2*log(2)) * abs(best[3])
fwhm_err = 2 * sqrt(2*log(2)) * abs(err[3])
s += '%-5s %.5f +/- %.5f'%('fwhm',fwhm,fwhm_err)

fig.text(.2,.7,s)
title('OceanOptics Spectra: ' + str(filename))
'''
