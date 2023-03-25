# GP_Graph_Net

The QM9 dataset has 11 features for each molecule, which are:

1. Molecular name
1. Rotational constant A
1. Rotational constant B
1. Rotational constant C
1. Dipole moment
1. Isotropic polarizability
1. HOMO energy
1. LUMO energy
1. Gap between HOMO and LUMO
1. Electronic spatial extent
1. Zero point vibrational energy (ZPVE)

Edges have values representing the number of bods between atoms
- (1, 2, 3, 4)

The 19 molecular properties or targets in the QM9 dataset are:

1. mu: Dipole moment (in Debye)
1. alpha: Isotropic polarizability (in Bohr^3)
1. homo: Energy of the highest occupied molecular orbital (in 1. Hartree)
1. lumo: Energy of the lowest unoccupied molecular orbital (in1.  1. Hartree)
1. gap: Gap, defined as lumo - homo (in Hartree)1. 
1. r2: Electronic spatial extent (in Bohr^2)1. 
1. zpve: Zero-point vibrational energy (in Hartree)1. 
1. U0: Internal energy at 0 K (in Hartree)1. 
1. U: Internal energy at 298.15 K (in Hartree)1. 
1. H: Enthalpy at 298.15 K (in Hartree)1. 
1. G: Free energy at 298.15 K (in Hartree)1. 
1. Cv: Heat capacity at 298.15 K (in cal/mol-K)1. 
1. omega1: Frequency of the lowest non-zero normal mode (in cm1. ^1. -1)
1. omega2: Frequency of the second lowest non-zero normal mode 1. (in cm^-1)
1. omega3: Frequency of the third lowest non-zero normal mode (in cm^-1)
1. zpe: Zero-point energy correction to U0 (in Hartree)
1. dipole: Total molecular dipole moment (in Debye)
1. polarizability: Anisotropic polarizability (in Bohr^3)
1. homo-Lumo gap: Gap, defined as lumo - homo (in eV)