import numpy as np

from SMT.core.physical_parameters import ThomasFermiZbar, Fermi_Energy
from SMT.core.physical_constants import *

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

    knm = np.where(g<1, -n/4 * np.math.factorial(m - 1) * np.log( np.sum(a[:,np.newaxis,np.newaxis]*g_arr,axis=0) )  , (b[0] + b[1]*np.log(g) + b[2]*np.log(g)**2)/(1 + b[3]*g + b[4]*g**2) )
    
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

	def __init__(self, N_ions, ion_masses_AU, Zion_array, T_array_AU, ni_array_AU, Zbar_array=None):
		"""
		Electrons....
		"""
		self.N_ions   = N_ions
		self._Z_array  = Zion_array 
		self._T_array  = T_array_AU
		self._mi_array  = ion_masses_AU
		self._ni_array = ni_array_AU
		self._Zbar_input = Zbar_array

		self.Zbar_array = np.zeros(self.N_ions)

		self.m_array = np.zeros(self.N_ions + 1 )
		self.n_array = np.zeros(self.N_ions + 1 )

		self.charge_matrix = np.ones((self.N_ions + 1 ,self.N_ions + 1 ))

		self.update_all_transport()

	@property
	def T_array(self):
		return self._T_array

	@T_array.setter
	def T_array(self, T_array_update):
		self._T_array = T_array_update
		self.update_all_transport()

	@property
	def ni_array(self):
		return self._ni_array

	@ni_array.setter
	def ni_array(self, n_array_update):
		self._ni_array = ni_array_update
		self.update_all_transport()
	
	@property
	def Zi_array(self):
		return self._Zi_array

	@Zi_array.setter
	def Zi_array(self, n_array_update):
		self._Zi_array = Zi_array_update
		self.update_all_transport()

	@property
	def mi_array(self):
		return self._mi_array

	@mi_array.setter
	def mi_array(self, m_array_update):
		self._mi_array   = mi_array_update
		self.update_all_transport()

	def update_K_nm(self):
		self.K_11_matrix = K_nm(self.g_matrix, 1, 1)
		self.K_12_matrix = K_nm(self.g_matrix, 1, 2)
		self.K_21_matrix = K_nm(self.g_matrix, 1, 1)
		self.K_22_matrix = K_nm(self.g_matrix, 2, 2)

	def update_number_densities(self):
		ne = np.sum(self.Zbar_array*self._ni_array)
		self.n_array[0]  = ne
		self.n_array[1:] = self._ni_array  
		self.x_array = self.n_array/np.sum(self.n_array)

	def update_Zbar(self):
		if self._Zbar_input is None:
			self.Zbar_array[:] = ThomasFermiZbar(self._Z_array, self._ni_array, self._T_array[1:])
		else:
			self.Zbar_array[:] = self._Zbar_input
		self.charge_matrix[0,1:]  = + self.Zbar_array
		self.charge_matrix[1:,0]  = + self.Zbar_array
		self.charge_matrix[1:,1:] = self.Zbar_array[np.newaxis,:]*self.Zbar_array[:,np.newaxis]

	def update_masses(self):
		self.m_array[0]  = m_e
		self.m_array[1:] = self._mi_array
		self.m_matrix = 2*self.m_array[:,np.newaxis]*self.m_array[np.newaxis,:]/(self.m_array[:,np.newaxis]+self.m_array[np.newaxis,:]) # twice the reduced mass

	def update_T_matrix(self):
		self.T_matrix = (self.m_array[:,np.newaxis]*self.T_array[np.newaxis,:] +
						 self.m_array[np.newaxis,:]*self.T_array[:,np.newaxis])/(self.m_array[:,np.newaxis]+self.m_array[np.newaxis,:])
		self.β_matrix = 1/self.T_matrix

	def update_screening(self):
		Ti = self.T_array[1:]
		ρion = np.sum( self.ni_array*self.Zbar_array  )
		ri_eff = (3*self.Zbar_array/ (4*π*ρion) )**(1/3)
		ΓISi = Ti*self.Zbar_array**2/ri_eff 

		ne = self.n_array[0]
		Te = self.T_array[0]
		EF = Fermi_Energy(ne)
		λe = 1/np.sqrt(4*π*ne/(Te**(9/5) + (2/3*EF)**(9/5)  )**(5/9) ) 
		λi = 1/np.sqrt(4*π*self.Zbar_array**2*ne/self.T_array[1:]) 
		self.λeff = 1/np.sqrt( 1/λe**2 + np.sum(1/λi**2*(1/(1+3*ΓISi)))  )
		self.g_matrix = self.β_matrix*self.charge_matrix/self.λeff

	def update_physical_params(self):
		self.update_masses()
		self.update_T_matrix()
		self.update_Zbar()
		self.update_number_densities()
		self.update_screening()		

	def update_all_transport(self):
		"""
		updates everything
		"""		
		self.update_physical_params()
		self.update_K_nm()
		self.inter_diffusion()
		self.electrical_conductivity()

	def inter_diffusion(self):
		""" Computes the interdiffusion coefficient using generalization of eq. 12 from [2].

		Returns
		-------
		Dei : matrix
		    Self-diffusion coefficients for system parameters.
		    
		"""
		if (self.T_array != np.ones_like(self.T_array)*self.T_array[0]).all():
			print("Warning: Multiple temperature interdiffusion not implemented! Assuming temperature is cross temeprature.")
		
		nij = self.n_array[np.newaxis,:] + self.n_array[:,np.newaxis]
		Zij = self.charge_matrix
		self.Dij = 3*self.T_matrix**(5/2)/(16*np.sqrt(np.pi*self.m_matrix)*nij*Zij**2*self.K_11_matrix)
		return self.Dij
 	
	def electrical_conductivity(self):
		""" Computes the interdiffusion coefficient using generalization of eq. 13 from [2].

		Returns
		-------
		σ : float
			Self-diffusion coefficients for system parameters.
		    
		"""
		Dei = self.Dij[0,1:]
		xi = self.x_array[1:]
		Ti = self.T_array[1:]
		self.σ = self.n_array[0]*np.sum( Ti*xi/Dei )
		return self.σ









