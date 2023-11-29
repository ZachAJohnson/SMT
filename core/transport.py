import numpy as np


def K_nm(g, n, m):
    """ Computes the plasma parameters (e.g. ion plasma frequency, ion-sphere radius, coupling parameter, etc.).
                                    
    Parameters
    ----------
    g : float or array_like
        Plasma parameter (eq. 54 from [1]) 
    n : int
        Subscript for collision intergral Knm (eq. C22 from [1])
    m : int
        Subscript for collision integral Knm (eq. C22 from [1])
        
    Returns
    -------
    knm : array_like
        Fit to collision integral (eqs. C22-C24 from [1])
    """

    if n and m == 1:
        a = np.array([1.4660, -1.7836, 1.4313, -0.55833, 0.061162])
        b = np.array([0.081033, -0.091336, 0.051760, -0.50026, 0.17044])
        
    if n and m == 2:
        a = np.array([0.85401, -0.22898, -0.60059, 0.80591, -0.30555])
        b = np.array([0.43475, -0.21147, 0.11116, 0.19665, 0.15195])
        
    
    g_arr = np.array([g, g**2, g**3, g**4, g**5])

    if g < 1:
        knm = -n/4 * np.math.factorial(m - 1) * np.log( np.dot(a,g_arr) ) 
    else:
        knm = (b[0] + b[1]*np.log(g) + b[2]*np.log(g)**2)/(1 + b[3]*g + b[4]*g**2)

    return knm


class TransportProperties():
	"""Generate the Stanton and Murillo transport coefficients. test

	Args:


	References
	----------
	.. [1] `Stanton, Liam G., and Michael S. Murillo. "Ionic transport in high-energy-density matter."
	 Physical Review E 93.4 (2016): 043203. <https://doi.org/10.1103/PhysRevE.93.043203>`_
	   [2] `Stanton, Liam G., and Michael S. Murillo.  ""Efficient model for electronic transport in high energy-
		density matter" Phys. Plasmas 28, 082301 (2021); <https://doi.org/10.1063/5.0048162>

	"""

	def __init__(self, N_ions, ion_masses_AU, Zion_array_AU, T_array_AU, n_array_AU, Zbar_array_AU=None):
		"""
		Electrons....
		"""
		self._N_ions  = N_ions
		self._m_array = ion_masses_AU
		self._T_array = T_array_AU
		self._n_array = n_array_AU

		self.update_all_transport()

	@property
	def T_array(self):
		return self._T_array

	@T_array.setter
	def T_array(self, T_array_update):
		self._T_array = T_array_update
		self.update_all_transport()

	@property
	def n_array(self):
		return self._n_array

	@n_array.setter
	def n_array(self, n_array_update):
		self._n_array = n_array_update
		self.update_all_transport()

	@property
	def m_array(self):
		return self._m_array

	@m_array.setter
	def m_array(self, m_array_update):
		self._m_array = m_array_update
		self.update_all_transport()

	def update_K_nm(self):
		self.K_11_matrix = K_nm(self.g_matrix, 1, 1)
		self.K_12_matrix = K_nm(self.g_matrix, 1, 2)
		self.K_21_matrix = K_nm(self.g_matrix, 1, 1)
		self.K_22_matrix = K_nm(self.g_matrix, 2, 2)

	def update_all_transport(self):
		"""
		updates everything
		"""
		self.g_matrix = np.diag(self.T_array)
		self.update_K_nm()

	# def self_diffusion(self):
	# 	""" Computes the self-diffusion coefficient using eq. 56 from [1].

	# 	Returns
	# 	-------
	# 	D : float or array_like
	# 	    Self-diffusion coefficients for system parameters.
		    
	# 	"""
	# 	D = np.sqrt(3*np.pi)/(12 * self.gamma**(5/2) * self.knm(self.g, 1, 1))
 	# 	return D
 	
 	def thermal_conductivities








