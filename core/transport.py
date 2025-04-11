import numpy as np

from SMT.core.physical_parameters import ThomasFermiZbar, Fermi_Energy, Degeneracy_Parameter, xc_PDW_h, xc_YOT, thermal_deBroglie_wavelength, rs_from_n, Kappa, Ion_Plasma_Frequency
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

	if g<1:
		knm = -n/4 * np.math.factorial(m - 1) * np.log( np.sum(a*g_arr,axis=0) )
	else:
		knm = (b[0] + b[1]*np.log(g) + b[2]*np.log(g)**2)/(1 + b[3]*g + b[4]*g**2) 
	return knm

class TransportProperties():
	"""Generate the Stanton and Murillo transport coefficients. Also included are (I)YVM viscosities, explicitly named.

	References
	----------
	.. [1] `Stanton, Liam G., and Michael S. Murillo. "Ionic transport in high-energy-density matter."
	 Physical Review E 93.4 (2016): 043203. <https://doi.org/10.1103/PhysRevE.93.043203>`_
	   [2] `Stanton, Liam G., and Michael S. Murillo.  ""Efficient model for electronic transport in high energy-
		density matter" Phys. Plasmas 28, 082301 (2021); <https://doi.org/10.1063/5.0048162>
	   [3] 'Johnson, Zach A., ....... (2024)'
	   [4] 'F. Perrot and M. Dharma-Wardana', Exchange and correlation potentials for electron-ion systems at finite tem-
      peratures, Physical Review A 30, 2619 (1984) <https://doi.org/10.1103/PhysRevA.30.2619>
       [5] 'Murillo, Michael  "Viscosity estimates of liquid metals and warm dense matter using the Yukawa reference systems."
        High Energy Density Physics, Volume 4, Issues 1–2, 2008, Pages 49-57, <https://doi.org/10.1016/j.hedp.2007.11.001.>
	"""

	def __init__(self, N_ions, ion_masses_AU, Zion_array, T_array_AU, ni_array_AU, Zbar_type='TF', Zbar_array=None,
						 improved_xc_SMT=False, improved_λdB_SMT=False, improved_ae_SMT=False, improved_PauliBlocking=False,  xc_type='YOT', λdB_n = 2):
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
	def ni_array(self, ni_array_update):
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
		self.f_PB_11_matrix = np.ones_like(self.charge_matrix)	
		self.f_PB_12_matrix = np.ones_like(self.charge_matrix)	
		self.f_PB_22_matrix = np.ones_like(self.charge_matrix)	
		self.f_PB_13_matrix = np.ones_like(self.charge_matrix)	
		
		if self.improved_PauliBlocking == True:
			θ = Degeneracy_Parameter(self.Te, self.ne)
			
			# First e-i correction
			f_PBei_function = lambda a, b, n: 1/(b*(1 + (a/θ)**n)**(1/n) )

			a_ei_11,b_ei_11,n_ei_11 = 3.009e-01,1.102e+00,1.788e+00 # K_1,1
			a_ei_12,b_ei_12,n_ei_12 = 2.361e-01,1.054e+00,1.864e+00 # K_1,2
			a_ei_22,b_ei_22,n_ei_22 = 2.361e-01,1.054e+00,1.864e+00 # K_2,2
			a_ei_13,b_ei_13,n_ei_13 = 1.940e-01,1.029e+00,1.969e+00 # K_1,3

			f_PBei_11 = f_PBei_function(a_ei_11, b_ei_11, n_ei_11)
			f_PBei_12 = f_PBei_function(a_ei_12, b_ei_12, n_ei_12)
			f_PBei_22 = f_PBei_function(a_ei_22, b_ei_22, n_ei_22)
			f_PBei_13 = f_PBei_function(a_ei_13, b_ei_13, n_ei_13)

			self.f_PB_11_matrix[0,1:] = f_PBei_11
			self.f_PB_12_matrix[0,1:] = f_PBei_12
			self.f_PB_22_matrix[0,1:] = f_PBei_22
			self.f_PB_13_matrix[0,1:] = f_PBei_13

			self.f_PB_11_matrix[1:,0] = f_PBei_11
			self.f_PB_12_matrix[1:,0] = f_PBei_12
			self.f_PB_22_matrix[1:,0] = f_PBei_22
			self.f_PB_13_matrix[1:,0] = f_PBei_13
			
			# Now e-e correction
			f_PBee_function = lambda a, b, n, m: 1/(b*(1 + (a/θ)**(m*n))**(1/n) )

			a_ee_11,b_ee_11,n_ee_11,m_ee_11 = 7.702e-01,1.500e+00,2.254e+00,1.146e+00 # K_1,1
			a_ee_12,b_ee_12,n_ee_12,m_ee_12 = 6.066e-01,1.331e+00,2.076e+00,1.172e+00 # K_1,2
			a_ee_22,b_ee_22,n_ee_22,m_ee_22 = 6.066e-01,1.331e+00,2.076e+00,1.172e+00 # K_2,2
			a_ee_13,b_ee_13,n_ee_13,m_ee_13 = 4.975e-01,1.225e+00,1.987e+00,1.196e+00 # K_1,3

			f_PBee_11 = f_PBee_function(a_ee_11, b_ee_11, n_ee_11, m_ee_11)
			f_PBee_12 = f_PBee_function(a_ee_12, b_ee_12, n_ee_12, m_ee_12)
			f_PBee_22 = f_PBee_function(a_ee_22, b_ee_22, n_ee_22, m_ee_22)
			f_PBee_13 = f_PBee_function(a_ee_13, b_ee_13, n_ee_13, m_ee_13)

			self.f_PB_11_matrix[0,0] = f_PBee_11
			self.f_PB_12_matrix[0,0] = f_PBee_12
			self.f_PB_22_matrix[0,0] = f_PBee_22
			self.f_PB_13_matrix[0,0] = f_PBee_13

			self.f_PB_11_matrix[0,0] = f_PBee_11
			self.f_PB_12_matrix[0,0] = f_PBee_12
			self.f_PB_22_matrix[0,0] = f_PBee_22
			self.f_PB_13_matrix[0,0] = f_PBee_13
		
		self.K_11_matrix = self.f_PB_11_matrix * K_nm(self.g_matrix, 1, 1)
		self.K_12_matrix = self.f_PB_12_matrix * K_nm(self.g_matrix, 1, 2)
		self.K_21_matrix = self.f_PB_12_matrix * K_nm(self.g_matrix, 2, 1)
		self.K_22_matrix = self.f_PB_22_matrix * K_nm(self.g_matrix, 2, 2)
		self.K_13_matrix = self.f_PB_13_matrix * K_nm(self.g_matrix, 1, 3)

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
			self._Zbar_array[:] = ThomasFermiZbar(self._Zi_array, self._ni_array, self._T_array[0])
		
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

	def f_dB_improved_SMT(self):
		"""
		Diffraction corrected 
		"""
		λdB_matrix = 1/np.sqrt(π*self.m_matrix*self.T_matrix) # As defined using Λij in [3], used in [3]
		# λdB_matrix = 1/np.sqrt(2*self.m_matrix*self.T_matrix) # As defined in GMS (by generalizing me-> μij = mij/2)
		# λdB_matrix = np.sqrt(2*π)/np.sqrt(0.5*self.m_matrix*self.T_matrix) # Wikipedia with me -> mij/2

		rc_matrix = np.abs(self.charge_matrix)/self.T_matrix		

		self.f_dB_matrix = 1 + λdB_matrix**2/rc_matrix**2
	
		
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
			self.f_dB_improved_SMT()
		else:
			self.f_dB_matrix = np.ones_like(self.charge_matrix)

		if self.improved_ae_SMT == True: # Low T Clamp on screening length by ae
			self.ae = rs_from_n(self.ne)
			self.ae_correction = self.ae
		else:
			self.ae_correction = 0
	
		# Array of ionic screening lengths 
		self.λi_array = 1/np.sqrt(4*π*self._Zbar_array**2*self.ni_array/self.Ti_array) 	

		self.λeff = 1/np.sqrt( 1/(self.λe**2 + self.ae_correction**2 - self.γ0 ) + np.sum( 1/(self.λi_array**2 + self.ri_eff**2)  ))
		self.g_matrix = self.β_matrix*self.charge_matrix* np.sqrt(self.f_dB_matrix)/self.λeff # Equivalent to [3], but made not part of λeff here.

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
		self.thermal_conductivity()
		self.viscosity()
		self.YVM_viscosity()
		self.IYVM_viscosity()

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

	def thermal_conductivity(self):
		"""
		Thermal conductivity approximated by Eq.17 of [2]
		Note there is an ambiguity about how to divide the conductivity between electronic and ionic contributions, resulting in the various κee, κii, κe, and κe_no_ee definitions below.  
		"""
		if self.N_ions > 1:
				print("Warning about themal conductivity: Only single-ion implemented. Returns array of single-species e-i conductivities of each species input.")		

		Tei = self.T_matrix[0,1:]
		Te  = self.T_matrix[0,0]
		Λ   = np.sqrt(8)*self.K_22_matrix[0,0] + self._Zbar_array*(25*self.K_11_matrix[0,1:] - 20*self.K_12_matrix[0,1:] + 4*self.K_13_matrix[0,1:]) 
		self.κ   = 75*Tei**2.5/(16*np.sqrt(2*π*m_e)*Λ)
		# self.κii = 75*self.Ti_array**2.5/(64*np.sqrt(π*self.mi_array)*np.diag(self.charge_matrix)[1:]**0*np.diag(self.K_22_matrix)[1:] )
		# self.κii = 75*self.Ti_array**2.5/(64*np.sqrt(π*self.mi_array)*np.diag(self.charge_matrix)[1:]**2*np.diag(self.K_22_matrix)[1:] )
		Ωii_22 = np.diag(self.Ω_22_matrix)[1:]
		self.κii = 75*self.Ti_array/(32*Ωii_22*self.mi_array)
		self.κee = 75*Te**2.5/(64*np.sqrt(π*m_e)*self.K_22_matrix[0,0])
		self.κe  = self.κ

		# Define κe by removing Kii, Kee
		# Λei   = self._Zbar_array*(25*self.K_11_matrix[0,1:] - 20*self.K_12_matrix[0,1:] + 4*self.K_13_matrix[0,1:]) 
		# self.κe_no_ee  = 75*Tei**2.5/(16*np.sqrt(2*π*m_e)*Λei)
		self.κe_no_ee  = self.κ/(1-self.κ/self.κee)
		
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

	def YVM_viscosity(self):
		"""
		The viscosity based on the YVM model [5], itself based on Yukawa molecular dynamic viscosities.
		Only single species implemented so far. Output is array of each ionic viscosity.
		"""
		if self.N_ions > 1:
			print("Warning about themal conductivity: Only single-ion implemented. Returns array of single-species e-i conductivities of each species input.")		

		κ_array = Kappa(self.T_array[0], self.ni_array, self.Zbar_array)

		ω_p_array = Ion_Plasma_Frequency(self.ni_array, self.mi_array, self.Zbar_array)
		ω_E_array = np.exp(-0.2*κ_array**1.62)/np.sqrt(3)*ω_p_array

		ai_array = rs_from_n(self.ni_array)
		η_0_array   = np.sqrt(3) * ω_E_array * self.mi_array * self.ni_array * ai_array**2
		Γ_array = self.Zbar_array**2/(ai_array*self.Ti_array)

		# Yukawa melting transition temperature
		self.Γ_melt_array = 171.8 + 82.8*(np.exp(0.565*κ_array**1.38) - 1) 

		# Actual YVM formula
		self.η_YVM_array = η_0_array*(0.0051*self.Γ_melt_array/Γ_array + 0.374*Γ_array/self.Γ_melt_array + 0.022)
		return self.η_YVM_array # * AU_to_g*AU_to_invcc*AU_to_cm**2/AU_to_s

	def IYVM_viscosity(self):
		"""
		Extension to YVM based on improvements in [4], which include high-T SMT results for the fit.
		"""
		if self.N_ions > 1:
			print("Warning about themal conductivity: Only single-ion implemented. Returns array of single-species e-i conductivities of each species input.")		

		κ_array = Kappa(self.T_array[0], self.ni_array, self.Zbar_array)

		ω_p_array = Ion_Plasma_Frequency(self.ni_array, self.mi_array, self.Zbar_array)
		ω_E_array = np.exp(-0.2*κ_array**1.62)/np.sqrt(3)*ω_p_array

		ai_array = rs_from_n(self.ni_array)
		η_0_array   = np.sqrt(3) * ω_E_array * self.mi_array * self.ni_array * ai_array**2
		Γ_array = self.Zbar_array**2/(ai_array*self.Ti_array)

		# Yukawa melting transition temperature
		self.Γ_melt_array = 171.8 + 82.8*(np.exp(0.565*κ_array**1.38) - 1) 

		# Now for numerical parameters for fit
		A = 1.45e-4 - 1.04e-4*κ_array + 3.69e-5*κ_array**2
		B = 0.3 + 0.86*κ_array-0.69*κ_array**2 + 0.138*κ_array**3
		C = 0.015 + 0.048*κ_array*0.754
		a = 1.78 + 0.13*κ_array - 0.062*κ_array**2
		b = 1.63 - 0.325*κ_array + 0.24*κ_array**2

		# Actual IYVM formula
		self.η_IYVM_array = η_0_array*( A*(self.Γ_melt_array/Γ_array)**a + B*(Γ_array/self.Γ_melt_array)**b + C )

		return self.η_IYVM_array # * AU_to_g*AU_to_invcc*AU_to_cm**2/AU_to_s
	
	def get_SMT_η_cgs(self, Te_AU, Ti_AU, Zbar=None):
		self.T_array = np.array([Te_AU, Ti_AU ])
		if Zbar is not None:
			self.Zbar_array = np.array([Zbar])
		print(self.charge_matrix**2)
		return self.ηi*AU_to_g*AU_to_invcc*AU_to_cm**2/AU_to_s

	def get_YVM_η_cgs(self, Te_AU, Ti_AU, Zbar=None):
		self.T_array = np.array([Te_AU, Ti_AU ])
		if Zbar is not None:
			self.Zbar_array = np.array([Zbar])
		return self.η_YVM_array*AU_to_g*AU_to_invcc*AU_to_cm**2/AU_to_s

	def get_IYVM_η_cgs(self, Te_AU, Ti_AU, Zbar=None):
		self.T_array = np.array([Te_AU, Ti_AU ])
		if Zbar is not None:
			self.Zbar_array = np.array([Zbar])
		return self.η_IYVM_array*AU_to_g*AU_to_invcc*AU_to_cm**2/AU_to_s

	def get_SMT_Zbar(self, Te_AU, Ti_AU, Zbar=None):
		self.T_array = np.array([Te_AU, Ti_AU ])
		if Zbar is not None:
			self.Zbar_array = np.array([Zbar])
		return self.Zbar_array

	def get_SMT_κi_cgs(self, Te_AU, Ti_AU, Zbar=None):
		self.T_array = np.array([Te_AU, Ti_AU ])
		if Zbar is not None:
			self.Zbar_array = np.array([Zbar])
		return self.κii*AU_to_erg/(AU_to_cm*AU_to_s*AU_to_K)

	def get_SMT_D_cgs(self, Te_AU, Ti_AU, Zbar=None):
		self.T_array = np.array([Te_AU, Ti_AU ])
		if Zbar is not None:
			self.Zbar_array = np.array([Zbar])
		return self.D_array[1]*AU_to_cm**2/AU_to_s
