import pylab as plt

def plot_mainChamber(time,V,P,T,eps_x,eps_g,rho,crustal_density,pref='./') :
    plt.figure(1),
    plt.plot(time/(3600.*24.*365.),V/V[0])
    plt.xlabel('time (yr)')
    plt.ylabel('volume/V_0')
    plt.title('Magma Reservoir Volume Evolution')

    plt.figure(2) #%,clf
    plt.plot(time/(3600.*24.*365.),P/1e6)
    plt.xlabel('time (yr)')
    plt.ylabel('pressure (MPa)')
    plt.title('Magma Reservoir Pressure Evolution')
    plt.savefig(pref+'P_val.pdf')
    #set(gca)

    plt.figure(3)
    plt.plot(time/(3600.*24.*365.),T)
    plt.xlabel('time (yr)')
    plt.ylabel('temperature (K)')
    plt.title('Magma Reservoir Temperature Evolution')
    plt.savefig(pref+'T_val.pdf')
    #set(gca)

    plt.figure(4)
    plt.plot(time/(3600.*24.*365.),eps_g)
    plt.xlabel('time (yr)')
    plt.ylabel('gas volume fraction fraction')
    plt.title('Magma Reservoir Gas Volume Fraction Evolution')
    plt.savefig(pref+'T_val.pdf')
    #set(gca)

    plt.figure(5)
    plt.plot(time/(3600.*24.*365.),eps_x)
    plt.xlabel('time (yr)')
    plt.ylabel('crystal volume fraction')
    plt.title('Magma Reservoir Crystal fraction Evolution')
    plt.savefig(pref+'T_val.pdf')
    #set(gca)

    plt.figure(6)
    plt.plot(time/(3600.*24.*365.),rho/crustal_density)
    plt.xlabel('time (yr)')
    plt.ylabel('mean density/crustal_density')
    plt.title('Magma Reservoir density anomaly Evolution')
    plt.savefig(pref+'T_val.pdf')
    #set(gca)
    #plt.show()
