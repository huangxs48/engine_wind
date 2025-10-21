import numpy as np 

#feel free to replace these constant by astropy.unit 
m_p = 1.6726e-24
k_B = 1.3807e-16
Lsun = 3.85e33
Msun = 1.9891e33
Rsun = 6.955e10
year = 3.155815e7
pc = 3.0857e18
c_light = 2.99792458e10
G = 6.67259e-8
AU = 1.496e13
a_Rad = 7.5657e-15
gas_gamma = 5.0/3.0

def scinote(x):
	return "{:e}".format(x)

class code_units:
    def __init__(self, rho_unit, temp_unit, l_unit, mmw=0.6):
        #default unit of length=rg, velocity=c_light, density=density_unit
        #self.vel_unit = vel_unit
        self.mmw = mmw
        self.rho_unit = rho_unit
        self.l_unit = l_unit
        self.temp_unit = temp_unit

        #derived units
        #temperature unit
        R_ideal = k_B/mmw/m_p
        #!!there's no gas_gamma in temperature unit, see Athena++'s wiki
        #becasue the code is solved with P'=rho'*T'
        self.mass_unit = self.rho_unit*pow(self.l_unit, 3)
        #self.temp_unit = self.vel_unit*self.vel_unit/R_ideal
        self.vel_unit = np.sqrt(self.temp_unit*R_ideal)
        self.time_unit = self.l_unit/self.vel_unit
        self.prat = a_Rad*pow(self.temp_unit, 4)/(self.rho_unit*pow(self.vel_unit, 2))
        self.crat = c_light/self.vel_unit
        self.mass_unit = self.rho_unit*pow(self.l_unit, 3)
        self.mdot_unit = self.mass_unit/self.time_unit
        self.press_unit = self.rho_unit*pow(self.vel_unit, 2)

    def print_unit(self):
        print('--- chosen units ---')
        print("dens unit:", scinote(self.rho_unit), "g/cm^3")
        print("length unit:", scinote(self.l_unit), "cm, also", self.l_unit/Rsun, "R_sun")
        print("temp unit: ", scinote(self.temp_unit), "K")
        
        print()
        print("--- radiation block of input file ---")
        print("prat = ", self.prat)
        print("crat = ", self.crat)

        print()
        print("--- dedrived units ---")
        print("velocity unit:", scinote(self.vel_unit), "cm/s, also", self.vel_unit/c_light, "c")
        print("time unit:", self.time_unit, "s")
        print("mass unit:", scinote(self.mass_unit), "g, also", self.mass_unit/Msun, "M_sun")
        print("mdot unit:", self.mass_unit/self.time_unit, \
	           "g/s, also", scinote((self.mass_unit/Msun)/(self.time_unit/3600./24./365.)), "Msun/yr")
        print("AM unit:", scinote(self.vel_unit*self.l_unit), \
	           "cm^2/s, also", scinote((self.vel_unit/c_light)*(self.l_unit/Rsun)), "c*Rsun")

    print()


# u = code_units(rho_unit=1.0e-10, temp_unit=1.0e5, l_unit=1.0e12, mmw=0.6)
# u.print_unit()

# #example usage
# print("one day is: ", 24*3600/u.time_unit)
# print("one hour is: ", 3600/u.time_unit)

# #example utilities
# def t_visc(mbh, r_orb, alpha=0.1, hr=0.1):
#     P_orb = 2.0*np.pi/np.sqrt(G*mbh / pow(r_orb, 3))
#     return P_orb * pow(hr, -2) * pow(alpha, -1)

# def t_diff(mbh, r_diff, kappaes=0.34, mfb=0.16*Msun):
#     return 3*kappaes*mfb/(4.0*np.pi*r_diff*c_light)

# def p_orb(mbh, r_orb):
#     return 2.0*np.pi/np.sqrt(G*mbh / pow(r_orb, 3))



