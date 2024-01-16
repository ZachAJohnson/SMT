import numpy as np

from SMT.core.physical_parameters import ThomasFermiZbar, Fermi_Energy, Degeneracy_Parameter, xc_PDW_h, xc_YOT, thermal_deBroglie_wavelength, rs_from_n
from SMT.core.physical_constants import *

@np.vectorize
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

	if n==1 and m == 1:
		a = np.array([1.4660, -1.7836, 1.4313, -0.55833, 0.061162])
		b = np.array([0.081033, -0.091336, 0.051760, -0.50026, 0.17044])

	if n==2 and m == 2:
		a = np.array([0.85401, -0.22898, -0.60059, 0.80591, -0.30555])
		b = np.array([0.43475, -0.21147, 0.11116, 0.19665, 0.15195])

	if (n==1 and m==2) or (m==1 and n==2):
		a = np.array([0.52094, 0.25153, -1.1337, 1.2155, -0.43784])
		b = np.array([0.20572, -0.16536, 0.061572, -0.12770, 0.066993])

	if (n==1 and m==3) or (n==3 and m==1):
		a = np.array([0.30346, 0.23739, -0.62167, 0.56110, -0.18046])
		b = np.array([0.68375, -0.38459, 0.10711, 0.10649, 0.028760]) 

	g_arr = np.array([g, g**2, g**3, g**4, g**5])

	# knm = np.where( g<1, 
	# 			    -n/4 * np.math.factorial(m - 1) * np.log( np.sum(a[:,np.newaxis,np.newaxis]*g_arr,axis=0) ) ,
	# 				(b[0] + b[1]*np.log(g) + b[2]*np.log(g)**2)/(1 + b[3]*g + b[4]*g**2) 
	# 	    	  )    

	if g<1:
		knm = -n/4 * np.math.factorial(m - 1) * np.log( np.sum(a*g_arr,axis=0) )
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
	   [3] 'Johnson, Zach A., ....... (2024)'
	   [4] 'F. Perrot and M. Dharma-Wardana', Exchange and correlation potentials for electron-ion systems at finite tem-
      peratures, Physical Review A 30, 2619 (1984) <https://doi.org/10.1103/PhysRevA.30.2619>
	"""

	def __init__(self, N_ions, ion_masses_AU, Zion_array, T_array_AU, ni_array_AU, Zbar_type='TF', Zbar_array=None,
						 improved_xc_SMT=False, improved_λdB_SMT=False, improved_ae_SMT=False, improved_PauliBlocking=False,  xc_type='PDW', λdB_n = 2):
		"""
		Defaults to SMT model in [1,2], with improved_... denoting the improvements in [3]
		"""
		self.N_ions   = N_ions
		self._Zi_array  = Zion_array 
		self._T_array  = T_array_AU
		self._mi_array  = ion_masses_AU
		self._ni_array = ni_array_AU

		# Whether or not to include an xc correction to the screening length, see [3]
		self.improved_xc_SMT  = improved_xc_SMT
		self.improved_λdB_SMT = improved_λdB_SMT
		self.improved_ae_SMT  = improved_ae_SMT
		self.improved_PauliBlocking =  improved_PauliBlocking
		self._xc_type         = xc_type
		self._λdB_n           = λdB_n

		# Make either TF Zbar or use input Zbar
		self.Zbar_type = Zbar_type
		if (self.Zbar_type == 'input') and (Zbar_array is not None):	
			self._Zbar_array = Zbar_array
		else:
			self._Zbar_array = np.zeros(self.N_ions)
		
		self.m_array = np.zeros(self.N_ions + 1 )
		self.n_array = np.zeros(self.N_ions + 1 )

		self.charge_matrix = np.ones((self.N_ions + 1 ,self.N_ions + 1 ))

		self.update_all_params()

	@property
	def T_array(self):
		return self._T_array

	@T_array.setter
	def T_array(self, T_array_update):
		self._T_array = T_array_update
		self.update_all_params()

	@property
	def ni_array(self):
		return self._ni_array

	@ni_array.setter
	def ni_array(self, n_array_update):
		self._ni_array = ni_array_update
		self.update_all_params()
	
	@property
	def Zi_array(self):
		return self._Zi_array

	@Zi_array.setter
	def Zi_array(self, Zi_array_update):
		self._Zi_array = Zi_array_update
		self.update_all_params()

	@property
	def Zbar_array(self):
		return self._Zbar_array

	@Zbar_array.setter
	def Zbar_array(self, Zbar_array_update):
		self._Zbar_array = Zbar_array_update
		self.update_all_params()

	@property
	def mi_array(self):
		return self._mi_array

	@mi_array.setter
	def mi_array(self, m_array_update):
		self._mi_array   = mi_array_update
		self.update_all_params()

	@property
	def xc_type(self):
		return self._xc_type

	@xc_type.setter
	def xc_type(self, xc_type):
		self._xc_type   = xc_type
		self.update_all_params()

	@property
	def λdB_n(self):
		return self._λdB_n

	@λdB_n.setter
	def λdB_n(self, λdB_n):
		self._λdB_n   = λdB_n
		self.update_all_params()

	def update_K_nm(self):
		if self.improved_PauliBlocking == True:
			θ = Degeneracy_Parameter(self.Te, self.ne)
			a, b, n = 0.52035809, 1.2766832,  1.83874532
			self.f_PB = 1/(b*(1 + (a/θ)**n)**(1/n) )
		else:
			self.f_PB = 1
		self.K_11_matrix = self.f_PB * K_nm(self.g_matrix, 1, 1)
		self.K_12_matrix = self.f_PB * K_nm(self.g_matrix, 1, 2)
		self.K_21_matrix = self.f_PB * K_nm(self.g_matrix, 2, 1)
		self.K_22_matrix = self.f_PB * K_nm(self.g_matrix, 2, 2)
		self.K_13_matrix = self.f_PB * K_nm(self.g_matrix, 1, 3)

	def update_collision_Ωij(self):
		self.Ω_11_matrix = np.sqrt(2*π/self.μ_matrix)*self.charge_matrix**2/self.T_matrix**1.5 * self.K_11_matrix
		self.Ω_12_matrix = np.sqrt(2*π/self.μ_matrix)*self.charge_matrix**2/self.T_matrix**1.5 * self.K_12_matrix
		self.Ω_21_matrix = np.sqrt(2*π/self.μ_matrix)*self.charge_matrix**2/self.T_matrix**1.5 * self.K_21_matrix
		self.Ω_22_matrix = np.sqrt(2*π/self.μ_matrix)*self.charge_matrix**2/self.T_matrix**1.5 * self.K_22_matrix
		self.Ω_13_matrix = np.sqrt(2*π/self.μ_matrix)*self.charge_matrix**2/self.T_matrix**1.5 * self.K_13_matrix

	def update_number_densities(self):
		self.ne = np.sum(self._Zbar_array*self._ni_array)
		self.n_array[0]  = self.ne
		self.n_array[1:] = self._ni_array  
		self.x_array = self.n_array/np.sum(self.n_array)

	def update_charges(self):
		# Check if need to compute TF ionization
		if self.Zbar_type == 'TF':
			self._Zbar_array[:] = ThomasFermiZbar(self._Zi_array, self._ni_array, self._T_array[1:])
		
		# Get full charge matrix Z_i Z_j
		self.charge_matrix[0,1:]  = self._Zbar_array
		self.charge_matrix[1:,0]  = self._Zbar_array
		self.charge_matrix[1:,1:] = self._Zbar_array[np.newaxis,:]*self._Zbar_array[:,np.newaxis]

	def update_masses(self):
		self.m_array[0]  = m_e
		self.m_array[1:] = self._mi_array
		self.m_matrix = 2*self.m_array[:,np.newaxis]*self.m_array[np.newaxis,:]/(self.m_array[:,np.newaxis]+self.m_array[np.newaxis,:]) # twice the reduced mass
		self.μ_matrix = self.m_matrix/2

	def update_T_matrix(self):
		self.Te = self.T_array[0]
		self.Ti_array = self.T_array[1:]
		self.T_matrix = (self.m_array[:,np.newaxis]*self.T_array[np.newaxis,:] +
						 self.m_array[np.newaxis,:]*self.T_array[:,np.newaxis])/(self.m_array[:,np.newaxis]+self.m_array[np.newaxis,:])
		self.β_matrix = 1/self.T_matrix
		

	def update_xc_correction(self):

		if self._xc_type == 'PDW':
			θ = Degeneracy_Parameter(self.Te, self.ne)

			h = xc_PDW_h(θ)
			hprime = (xc_PDW_h(θ*(1+1e-6)) - xc_PDW_h(θ*(1-1e-6)) )/(2e-6*θ)

			self.γ0 = θ/(8*self.Te) * (  h  - 2*θ*hprime )
		elif self._xc_type == 'YOT':
			self.γ0 = xc_YOT(self.Te, self.ne)
		else:
			print(f"WARNING: {self._xc_type} not implemented.")


	def standard_SMT_λe(self, option=1):
		"""
		Electron Screening length from original SMT [1,2]
		"""
		if option==1:
			self.λe = 1/np.sqrt(4*π*self.ne/(self.Te**(9/5) + (2/3*self.EF)**(9/5)  )**(5/9) ) # Option 1 approximation
		elif option==2:
			self.λe = 1/np.sqrt(4*π*self.ne/np.sqrt(self.Te**2 + (2/3*self.EF)**2  )) # Option 2 approximation

	def dB_improved_SMT_λe(self):
		"""
		Diffraction corrected 
		"""
		λdBroglie = thermal_deBroglie_wavelength(self.Te, m_e)
		xi_array = self.ni_array/np.sum(self.ni_array) 
		rc_av =   np.sum(xi_array * self.Zbar_array/ self.T_matrix[1:])/self.N_ions
		
		self.f_dB = (  1  +  (2*π*λdBroglie/rc_av)**2) # Does something
		# self.f_dB = (  1  +  (λdBroglie/rc_av)**2) # Does nothing

		# # Delete later
		# self.f_dB =  (2*π*λdBroglie/rc_av)**2 # Does something

	def update_screening(self):
		ρion = np.sum( self.ni_array*self._Zbar_array  )
		self.ri_eff = (3*self._Zbar_array/ (4*π*ρion) )**(1/3)
		self.Γii_array = self._Zbar_array**2/(self.ri_eff*self.Ti_array) 

		self.EF = Fermi_Energy(self.ne)
		
		# Electron Screening length from original SMT (no xc)
		self.standard_SMT_λe()

		# Now add modifications
		self.f_mod_SMT = 1
		if self.improved_xc_SMT == True: #xc correction
			self.update_xc_correction()
		else:
			self.γ0 = 0

		if self.improved_λdB_SMT == True: # High T Quantum Dispersion Improvement
			self.dB_improved_SMT_λe()
		else:
			self.f_dB = 1

		if self.improved_ae_SMT == True: # Low T Clamp on screening length by ae
			self.ae = rs_from_n(self.ne)
			self.ae_correction = self.ae
		else:
			self.ae_correction = 0
	
		# Array of ionic screening lengths 
		self.λi_array = 1/np.sqrt(4*π*self._Zbar_array**2*self.ni_array/self.Ti_array) 	

		self.λeff = 1/np.sqrt( 1/(self.λe**2 + self.ae_correction**2 - self.γ0 ) + np.sum( 1/(self.λi_array**2 + self.ri_eff**2)  ))
		
		self.g_matrix = self.β_matrix*self.charge_matrix/self.λeff

	def update_physical_params(self):
		self.update_masses()
		self.update_T_matrix()
		self.update_charges()
		self.update_number_densities()
		self.update_xc_correction()
		self.update_screening()		

	def update_all_params(self):
		"""
		updates everything
		"""		
		self.update_physical_params()
		self.update_K_nm()
		self.update_collision_Ωij()
		self.inter_diffusion()
		self.electrical_conductivity()
		self.temperature_relaxation()
		self.viscosity()
		self.thermal_conductivity()

	def inter_diffusion(self):
		""" 
		Computes the interdiffusion coefficient using generalization of eq. 12 from [2], also eq. 66 from [1]
		To get cgs: AU_to_cm**2/AU_to_s
		Returns
		-------
		Dei : matrix
		    Self-diffusion coefficients for system parameters.
		    
		"""
		if (self.T_array != np.ones_like(self.T_array)*self.T_array[0]).all():
			print("Warning: Multiple temperature interdiffusion not implemented! Assuming temperature is cross temeprature.")
		
		# nij = self.n_array[np.newaxis,:] + self.n_array[:,np.newaxis] - np.diag(self.n_array) # Generalization of Eq. 12 from [2] and 55 of [1]
		ntot   = np.sum(self.n_array)
		Zij = self.charge_matrix
		self.Dij = 3*self.T_matrix**(5/2)/(16*np.sqrt(np.pi*self.m_matrix)* ntot *Zij**2*self.K_11_matrix)
		self.self_diffusion()
		return self.Dij

	def self_diffusion(self):
		"""
		Self diffusion defined as Dii * ntot/ni
		To get cgs: AU_to_cm**2/AU_to_s
		"""
		self.D_array = np.diag(self.Dij)*np.sum(self.n_array)/self.n_array
		return self.D_array
 	
	def electrical_conductivity(self):
		""" 
		Computes the conductivity using eq. 13 from [2].

		Returns
		-------
		σ : float
			Self-diffusion coefficients for system parameters.
		    
		"""
		Dei = self.Dij[0,1:]
		xi = self.x_array[1:]
		Ti = self.T_array[1:]
		self.σ = self.ne*np.sum( Ti*xi/Dei )**-1
		return self.σ

	def temperature_relaxation(self):
		"""
		Temperature relaxation between species, eq. 21 of [2]
		"""
		nj = self.n_array[np.newaxis,:]
		numerator = 3*(self.m_array[:, np.newaxis] + self.m_array[np.newaxis,:])*self.T_matrix**1.5
		denominator = (32*np.sqrt(π*self.m_matrix)*nj*self.charge_matrix**2*self.K_11_matrix )

		self.τij = numerator/denominator
		return self.τij

	def viscosity(self):
		"""
		Viscosity from Eq. 74 of [1] using multi-species Eq. 
		To get cgs: AU_to_g*AU_to_invcc*AU_to_cm**2/AU_to_s
		"""
		if self.N_ions > 1:
			print("Warning about viscosity: Only single-ion (and no electron) implemented. Returns single-species viscosity of each species input.")
		Ωii_22 = np.diag(self.Ω_22_matrix)[1:]
		self.ηi = 5*self.Ti_array/(8*Ωii_22)
		return self.ηi

	def thermal_conductivity(self):
		"""
		Thermal conductivity approximated by Eq.17 of [2]

		"""
		if self.N_ions > 1:
				print("Warning about themal conductivity: Only single-ion implemented. Returns array of single-species e-i conductivities of each species input.")		

		Tei = self.T_matrix[0,1:]
		Te  = self.T_matrix[0,0]
		Λ   = np.sqrt(8)*self.K_22_matrix[0,0] + self._Zbar_array*(25*self.K_11_matrix[0,1:] - 20*self.K_12_matrix[0,1:] + 4*self.K_13_matrix[0,1:]) 
		self.κ   = 75*Tei**2.5/(16*np.sqrt(2*π*m_e)*Λ)
		self.κii = 75*self.Ti_array**2.5/(64*np.sqrt(π*self.mi_array)* np.diag(self.K_22_matrix)[1:] )
		self.κee = 75*Te**2.5/(64*np.sqrt(π*m_e)*self.K_22_matrix[0,0])

		# Define κe by removing Kii, Kee
		# Λei   = self._Zbar_array*(25*self.K_11_matrix[0,1:] - 20*self.K_12_matrix[0,1:] + 4*self.K_13_matrix[0,1:]) 
		self.κe  = self.κ#75*Tei**2.5/(16*np.sqrt(2*π*m_e)*Λei)






