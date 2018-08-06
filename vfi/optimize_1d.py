import math
import numba

# constants
inv_phi = (math.sqrt(5) - 1) / 2 # 1/phi                                                                                                                     
inv_phi_sq = (3 - math.sqrt(5)) / 2 # 1/phi^2                                                                                                                  

# create
def create_optimizer(f):

	@numba.njit
	def golden_section_search(a,b,tol,*args):
		
		# a. distance
		dist = b - a
		if dist <= tol: 
			return (a+b)/2

		# b. number of iterations
		n = int(math.ceil(math.log(tol/dist)/math.log(inv_phi)))

		# c. potential new mid-points
		c = a + inv_phi_sq * dist
		d = a + inv_phi * dist
		yc = f(c,*args)
		yd = f(d,*args)

		# d. loop
		for _ in range(n-1):
			if yc < yd:
				b = d
				d = c
				yd = yc
				dist = inv_phi*dist
				c = a + inv_phi_sq * dist
				yc = f(c,*args)
			else:
				a = c
				c = d
				yc = yd
				dist = inv_phi*dist
				d = a + inv_phi * dist
				yd = f(d,*args)

		# e. return
		if yc < yd:
			return (a+d)/2
		else:
			return (c+b)/2
	
	return golden_section_search