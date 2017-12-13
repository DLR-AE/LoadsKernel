Die Software Loads Kernel ermoeglicht die Berechnung von quasi-stationären und dynamischen Manoeverlasten, instationaeren Boeenlasten im Zeitbereich, und dynamische Landlasten mit einem generischen Fahrwerksmodul im Zeitbereich. 

Aerodynamik:
- VLM (stationaer)
- DLM + RFA (instationaer)
- externe AICs (z.B. Nastran)
- AeroDB (Tau)
- CFD (Tau)
Struktur:
- Modalanalyse frei-frei
- Guyan Reduction
- externe Modes (z.B. Nastran SOL103)
- Voll- und Halbmodelle
EoM:
- 6 DoF, frei fliegend
- linear (aehnlich zu Nastran)
- nicht-linear (Waszak/Schmidt/Buttrill)
Aero-Struktur-Kopplung:
- rigid body spline
- surface spline
- volume spline

Quick Start Guide: https://phabricator.ae.go.dlr.de/w/loads_kernel_quick_start/
Manual: https://phabricator.ae.go.dlr.de/w/loads_kernel_manual/

Release_Gili_2017.10: https://phabricator.ae.go.dlr.de/diffusion/LAEK/repository/Release_Gili_2017.10/
Release_Bali_2016.10: https://phabricator.ae.go.dlr.de/diffusion/LAEK/repository/Release_Bali_2016.10/

