Dataset
===============

The dataset used in this repository is:

Duarte, Javier; (2019).
Sample with jet, track and secondary vertex properties for Hbb tagging ML studies HiggsToBBNTuple\_HiggsToBB\_QCD\_RunII\_13TeV\_MC.
CERN Open Data Portal.
https://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q

The dataset consists of particle jets extracted from simulated proton-proton collision events at a center-of-mass energy of 13 TeV generated with Pythia 8.
It has been produced for developing machine-learning algorithms to differentiate jets originating from a Higgs boson decaying to a bottom quark-antiquark pair (Hbb) from quark or gluon jets originating from quantum chromodynamic (QCD) multijet production.
The reconstructed jets are clustered using the anti-kT algorithm with R=0.8 from particle flow (PF) candidates (AK8 jets).
The standard L1+L2+L3+residual jet energy corrections are applied to the jets and pileup contamination is mitigated using the charged hadron subtraction (CHS) algorithm.
Features of the AK8 jets with transverse momentum pT > 200 GeV and pseudorapidity |η| < 2.4 are provided.
Selected features of inclusive (both charged and neutral) PF candidates with pT > 0.95 GeV associated to the AK8 jet are provided.
Additional features of charged PF candidates (formed primarily by a charged particle track) with pT > 0.95 GeV associated to the AK8 jet are also provided.
Finally, additional features of reconstructed secondary vertices (SVs) associated to the AK8 jet (within ∆R < 0.8) are also provided.

The features we use are as follows:
 ================================= ========= ==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
  Data variable                     Type      Description
 ================================= ========= ==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
  `sample_isQCD`                    Int_t     Boolean that is 1 if the simulated sample corresponds to QCD multijet production
  `fj_doubleb`                      Float_t   Double-b tagging discriminant based on a boosted decision tree calculated for the AK8 jet (see [CMS-BTV-16-002](http://cms-results.web.cern.ch/cms-results/public-results/publications/BTV-16-002/))
  `fj_eta`                          Float_t   Pseudorapidity η of the AK8 jet
  `fj_isH`                          Int_t     Boolean that is 1 if a generator-level Higgs boson and its daughters are geometrically matched to the AK8 jet
  `fj_isQCD`                        Int_t     Boolean that is 1 if none of the above matching criteria are satisfied (H, top, W, Z)
  `fj_jetNTracks`                   Float_t   Number of tracks associated with the AK8 jet
  `fj_nSV`                          Float_t   Number of SVs associated with the AK8 jet (∆R < 0.7)
  `fj_phi`                          Float_t   Azimuthal angle ϕ of the AK8 jet
  `fj_pt`                           Float_t   Transverse momentum of the AK8 jet
  `fj_sdmass`                       Float_t   Soft drop mass of the AK8 jet
  `fj_z_ratio`                      Float_t   z ratio variable as defined in [CMS-BTV-16-002](http://cms-results.web.cern.ch/cms-results/public-results/publications/BTV-16-002/)
  `fj_trackSipdSig_0`               Float_t   First largest track 3D signed impact parameter significance (see [CMS-BTV-16-002](http://cms-results.web.cern.ch/cms-results/public-results/publications/BTV-16-002/) )
  `fj_trackSipdSig_1`               Float_t   Second largest track 3D signed impact parameter significance (see [CMS-BTV-16-002](http://cms-results.web.cern.ch/cms-results/public-results/publications/BTV-16-002/) )
  `fj_trackSipdSig_2`               Float_t   Third largest track 3D signed impact parameter significance (see [CMS-BTV-16-002](http://cms-results.web.cern.ch/cms-results/public-results/publications/BTV-16-002/) )
  `fj_trackSipdSig_3`               Float_t   Fourth largest track 3D signed impact parameter significance (see [CMS-BTV-16-002](http://cms-results.web.cern.ch/cms-results/public-results/publications/BTV-16-002/) )
  `fj_trackSipdSig_0_0`             Float_t   First largest track 3D signed impact parameter significance associated to the first N-subjettiness axis
  `fj_trackSipdSig_0_1`             Float_t   Second largest track 3D signed impact parameter significance associated to the first N-subjettiness axis
  `fj_trackSipdSig_1_0`             Float_t   First largest track 3D signed impact parameter significance associated to the second N-subjettiness axis
  `fj_trackSipdSig_1_1`             Float_t   Second largest track 3D signed impact parameter significance associated to the second N-subjettiness axis
  `fj_trackSip2dSigAboveCharm_0`    Float_t   Track 2D signed impact parameter significance of the first track lifting the combined invariant mass of the tracks above the c hadron threshold mass (1.5 GeV)
  `fj_trackSip2dSigAboveBottom_0`   Float_t   Track 2D signed impact parameter significance of the first track lifting the combined invariant mass of the tracks above b hadron threshold mass (5.2 GeV)
  `fj_trackSip2dSigAboveBottom_1`   Float_t   Track 2D signed impact parameter significance of the second track lifting the combined invariant mass of the tracks above b hadron threshold mass (5.2 GeV)
  `fj_tau0_trackEtaRel_0`           Float_t   Smallest track pseudorapidity ∆η, relative to the jet axis, associated to the first N-subjettiness axis
  `fj_tau0_trackEtaRel_1`           Float_t   Second smallest track pseudorapidity ∆η, relative to the jet axis, associated to the first N-subjettiness axis
  `fj_tau0_trackEtaRel_2`           Float_t   Third smallest track pseudorapidity ∆η, relative to the jet axis, associated to the first N-subjettiness axis
  `fj_tau1_trackEtaRel_0`           Float_t   Smallest track pseudorapidity ∆η, relative to the jet axis, associated to the second N-subjettiness axis
  `fj_tau1_trackEtaRel_1`           Float_t   Second smallest track pseudorapidity ∆η, relative to the jet axis, associated to the second N-subjettiness axis
  `fj_tau1_trackEtaRel_2`           Float_t   Third smallest track pseudorapidity ∆η, relative to the jet axis, associated to the second N-subjettiness axis
  `fj_tau_vertexMass_0`             Float_t   Total SV mass for the first N-subjettiness axis, defined as the invariant mass of all tracks from SVs associated with the first N-subjettiness axis
  `fj_tau_vertexMass_1`             Float_t   Total SV mass for the second N-subjettiness axis, defined as the invariant mass of all tracks from SVs associated with the second N-subjettiness axis
  `fj_tau_vertexEnergyRatio_0`      Float_t   SV vertex energy ratio for the first N-subjettiness axis, defined as the total energy of all SVs associated with the first N-subjettiness axis divided by the total energy of all the tracks associated with the AK8 jet that are consistent with the PV
  `fj_tau_vertexEnergyRatio_1`      Float_t   SV energy ratio for the second N-subjettiness axis, defined as the total energy of all SVs associated with the first N-subjettiness axis divided by the total energy of all the tracks associated with the AK8 jet that are consistent with the PV
  `fj_tau_flightDistance2dSig_0`    Float_t   Transverse (2D) flight distance significance between the PV and the SV with the smallest uncertainty on the 3D flight distance associated to the first N-subjettiness axis
  `fj_tau_flightDistance2dSig_1`    Float_t   Transverse (2D) flight distance significance between the PV and the SV with the smallest uncertainty on the 3D flight distance associated to the second N-subjettiness axis
  `fj_tau_vertexDeltaR_0`           Float_t   Pseudoangular distance ∆R between the first N-subjettiness axis and SV direction
  `n_pfcands`                       Int_t     Number of particle flow (PF) candidates associated to the AK8 jet with transverse momentum greater than 0.95 GeV
  `npfcands`                        Float_t   Number of particle flow (PF) candidates associated to the AK8 jet with transverse momentum greater than 0.95 GeV
  `pfcand_deltaR`                   Float_t   Pseudoangular distance ∆R between the PF candidate and the AK8 jet axis
  `pfcand_drminsv`                  Float_t   Minimum pseudoangular distance ∆R between the associated SVs and the PF candidate
  `pfcand_drsubjet1`                Float_t   Pseudoangular distance ∆R between the PF candidate and the first soft drop subjet
  `pfcand_drsubjet2`                Float_t   Pseudoangular distance ∆R between the PF candidate and the second soft drop subjet
  `pfcand_erel`                     Float_t   Energy of the PF candidate divided by the energy of the AK8 jet
  `pfcand_etarel`                   Float_t   Pseudorapidity of the PF candidate relative to the AK8 jet axis
  `pfcand_phirel`                   Float_t   Azimuthal angular distance ∆ϕ between the PF candidate and the AK8 jet axis
  `pfcand_ptrel`                    Float_t   Transverse momentum of the PF candidate divided by the transverse momentum of the AK8 jet
  `pfcand_hcalFrac`                 Float_t   Fraction of energy of the PF candidate deposited in the hadron calorimeter
  `pfcand_puppiw`                   Float_t   Pileup per-particle identification (PUPPI) weight indicating whether the PF candidate is pileup-like (0) or not (1)
  `n_tracks`                        Int_t     Number of tracks associated with the AK8 jet
  `ntracks`                         Float_t   Number of tracks associated with the AK8 jet
  `trackBTag_DeltaR`                Float_t   Pseudoangular distance ∆R between the track and the AK8 jet axis
  `trackBTag_EtaRel`                Float_t   Pseudorapidity ∆η of the track relative the AK8 jet axis
  `trackBTag_JetDistVal`            Float_t   Minimum track approach distance to the AK8 jet axis
  `trackBTag_PParRatio`             Float_t   Component of track momentum parallel to the AK8 jet axis, normalized to the track momentum
  `trackBTag_PtRatio`               Float_t   Component of track momentum perpendicular to the AK8 jet axis, normalized to the track momentum
  `trackBTag_Sip2dVal`              Float_t   Transverse (2D) signed impact paramater of the track
  `trackBTag_Sip2dSig`              Float_t   Transverse (2D) signed impact paramater significance of the track
  `trackBTag_Sip3dSig`              Float_t   3D signed impact parameter significance of the track
  `trackBTag_Sip3dVal`              Float_t   3D signed impact parameter of the track
  `track_deltaR`                    Float_t   Pseudoangular distance (∆R) between the charged PF candidate and the AK8 jet axis
  `track_detadeta`                  Float_t   Track covariance matrix entry (eta, eta)
  `track_dlambdadz`                 Float_t   Track covariance matrix entry (lambda, dz)
  `track_dphidphi`                  Float_t   Track covariance matrix entry (phi, phi)
  `track_dphidxy`                   Float_t   Track covariance matrix entry (phi, xy)
  `track_dptdpt`                    Float_t   Track covariance matrix entry (pT, pT)
  `track_dxydxy`                    Float_t   Track covariance matrix entry (dxy, dxy)
  `track_dxydz`                     Float_t   Track covariance matrix entry (dxy, dz)
  `track_dzdz`                      Float_t   Track covariance matrix entry (dz, dz)
  `track_drminsv`                   Float_t   Minimum pseudoangular distance ∆R between the associated SVs and the charged PF candidate
  `track_dxy`                       Float_t   Transverse (2D) impact parameter of the track, defined as the distance of closest approach of the track trajectory to the beam line in the transverse plane to the beam
  `track_dxysig`                    Float_t   Transverse (2D) impact parameter significance of the track
  `track_dz`                        Float_t   Longitudinal impact parameter, defined as the distance of closest approach of the track trajectory to the PV projected on to the z direction
  `track_dzsig`                     Float_t   Longitudinal impact parameter significance of the track
  `track_erel`                      Float_t   Energy of the charged PF candidate divided by the energy of the AK8 jet
  `track_etarel`                    Float_t   Pseudorapidity ∆η of the track relative to the jet axis
  `track_mass`                      Float_t   Mass of the charged PF candidate
  `track_normchi2`                  Float_t   Normalized χ2 of the track fit
  `track_phirel`                    Float_t   Azimuthal angular distance ∆ϕ between the charged PF candidate and the AK8 jet axis
  `track_pt`                        Float_t   Transverse momentum of the charged PF candidate
  `track_ptrel`                     Float_t   Transverse momentum of the charged PF candidate divided by the transverse momentum of the AK8 jet
  `track_puppiw`                    Float_t   Pileup per-particle identification (PUPPI) weight indicating whether the PF candidate is pileup-like (0) or not (1)
  `track_quality`                   Float_t   Track quality: `undefQuality=-1`; `loose=0`; `tight=1`; `highPurity=2`; `confirmed=3`, if track found by more than one iteration; `looseSetWithPV=5`; `highPuritySetWithPV=6`, `discarded=7` if a better track found; `qualitySize=8`
  `n_sv`                            Int_t     Number of secondary vertices (SV) associated with the AK8 jet (∆R < 0.8)
  `nsv`                             Float_t   Number of secondary vertices (SV) associated with the AK8 jet (∆R < 0.8)
  `sv_chi2`                         Float_t   χ2 of the vertex fit
  `sv_normchi2`                     Float_t   χ2 divided by the number of degrees of freedom for the vertex fit
  `sv_costhetasvpv`                 Float_t   Cosine of the angle cos(θ) between the SV and the PV
  `sv_d3d`                          Float_t   3D flight distance of the SV
  `sv_d3derr`                       Float_t   3D flight distance uncertainty of the SV
  `sv_d3dsig`                       Float_t   3D flight distance significance of the SV
  `sv_dxy`                          Float_t   Transverse (2D) flight distance of the SV
  `sv_dxyerr`                       Float_t   Transverse (2D) flight distance uncertainty of the SV
  `sv_dxysig`                       Float_t   Transverse (2D) flight distance significance of the SV
  `sv_deltaR`                       Float_t   Pseudoangular distance ∆R between the SV and the AK8 jet
  `sv_erel`                         Float_t   Energy of the SV divided by the energy of the AK8 jet
  `sv_etarel`                       Float_t   Pseudorapidity ∆η of the SV relative to the AK8 jet axis
  `sv_mass`                         Float_t   Mass of the SV
  `sv_ntracks`                      Float_t   Number of tracks associated with the SV
  `sv_phirel`                       Float_t   Azimuthal angular distance ∆ϕ of the SV relative to the jet axis
  `sv_pt`                           Float_t   Transverse momentum of the SV
  `sv_ptrel`                        Float_t   Transverse momentum of the SV divided by the transverse momentum of the AK8 jet
 ================================= ========= ==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
