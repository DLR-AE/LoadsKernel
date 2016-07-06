function AJJ = ae_getaic(aerogrid, Mach, k)

% Modifications by Arne Voss (11/2015) to adapt to data structure of Loads Kernel. 
% Original function see below. The code was tested in octave 3.6.4. 
% However, runtime is slightly slower compared to Matlab 2013a. 

% Konvertierung von beliebigen IDs zu IDs 1:n
cornerpoint_old2new = [aerogrid.cornerpoint_grids(:,1), [1:length(aerogrid.cornerpoint_grids(:,1))]'];
for i =1:length(aerogrid.cornerpoint_panels(:,1))
    aerogrid.cornerpoint_panels(i,1) = cornerpoint_old2new( cornerpoint_old2new(:,1) == aerogrid.cornerpoint_panels(i,1) ,2);
    aerogrid.cornerpoint_panels(i,2) = cornerpoint_old2new( cornerpoint_old2new(:,1) == aerogrid.cornerpoint_panels(i,2) ,2);
    aerogrid.cornerpoint_panels(i,3) = cornerpoint_old2new( cornerpoint_old2new(:,1) == aerogrid.cornerpoint_panels(i,3) ,2);
    aerogrid.cornerpoint_panels(i,4) = cornerpoint_old2new( cornerpoint_old2new(:,1) == aerogrid.cornerpoint_panels(i,4) ,2);
end

Panel = [aerogrid.ID', aerogrid.cornerpoint_panels(:,[1,4,2,3])];
Node = [cornerpoint_old2new(:,2), aerogrid.cornerpoint_grids(:,2:end)];
S = aerogrid.A'; % panel areas
n_hat_w = aerogrid.N(:,3); % normal vector part in vertical direction
n_hat_wl = aerogrid.N(:,2); % normal vector part in lateral direction

% Folgende Zeile sind aus der Funktion getAIC kopiert, da die Funktionen getVLM und getDLM direkt aufgerufen werden.
% Dadurch werden unnoetige Aufrufe der Funktionen getVLM und getDLM vermieden und die Rechenzeit beschleunigt.

% determine number of aero panels present
[N,m]=size(Panel);

% rows of D are the downwash locations while columns define the effects of
% each aero panels doublet line/horseshoe vortex on that rows' downwash location

% define downwash location (3/4 chord and half span of the aero panel)
P0 = zeros(N,3);
P0(:,1) = (Node(Panel(:,2),2)+Node(Panel(:,3),2) + ...
        3*(Node(Panel(:,4),2)+Node(Panel(:,5),2)))/8; %xcp
P0(:,2) = (Node(Panel(:,2),3)+Node(Panel(:,3),3))/2; %ycp
P0(:,3) = (Node(Panel(:,2),4)+Node(Panel(:,3),4))/2; %zcp

% define doublet locations (1/4 chord and 0, half span and full span of the
% aero panel), kernel is computed at that 3 points and a parabolic function
% is fitted to approximate the kernel along the doublet line (ref 1,
% equation 7).
P1 = zeros(N,3);
P1(:,1) = Node(Panel(:,2),2)+ ...
         (Node(Panel(:,4),2) - Node(Panel(:,2),2))/4; %xp1
P1(:,2) = Node(Panel(:,2),3); %yp1
P1(:,3) = Node(Panel(:,2),4); %zp1

% P3 is doublet point at tip of the panel
P3 = zeros(N,3);
P3(:,1) = Node(Panel(:,3),2)+ ...
         (Node(Panel(:,5),2) - Node(Panel(:,3),2))/4; %xp3
P3(:,2) = Node(Panel(:,3),3); %yp3
P3(:,3) = Node(Panel(:,3),4); %zp3

% P2 is doublet point at the half-span location of the panel
P2 = zeros(N,3);
P2(:,1) = (P1(:,1)+P3(:,1))/2;
P2(:,2) = (P1(:,2)+P3(:,2))/2;
P2(:,3) = (P1(:,3)+P3(:,3))/2;

% define half span length and chord at centerline for each panel
s = (0.5*((Node(Panel(:,3),3) - Node(Panel(:,2),3))' + ...
          (Node(Panel(:,3),4) - Node(Panel(:,2),4))'));
c = (     (Node(Panel(:,4),2) - Node(Panel(:,2),2) + ...
           Node(Panel(:,5),2) - Node(Panel(:,3),2))/2)';

% get the downwash effect from DLM and VLM implementations for the defined geometry 

AJJ = zeros(length(Mach), length(k), size(Panel,1),size(Panel,1)); 
for im = 1:length(Mach)
    disp(['Ma = ',num2str(Mach(im))])
    Dv  = getVLM(P0,P1,P3,S,Mach(im),n_hat_w,n_hat_wl); % VLM (steady state effects)
    for ik = 1:length(k)
	if k(ik) == 0.0
	    Dd  = zeros(size(Panel,1),size(Panel,1)); % kein Anteil aus DLM, da steady state
	else
	    Dd  = getDLM(P0,P1,P2,P3,s,c,k(ik),Mach(im));           % DLM (oscillatory effects)
	end
        D   = Dv + Dd;
	AIC = -inv(D);
        AJJ(im,ik,:,:) = AIC;
    end
end

end

function [AIC] = getAIC(Panel,Node,Mach,k,S,n_hat_w,n_hat_wl)
% This function computes the AIC Matrix using ref. 1 as the primary
% reference for doublet lattice (DLM) implementation and ref 2 for vortex lattice   
% (VLM) implementation. The AIC matrix is a combination of DLM and VLM
%
% ref 1: Albano and Rodden - A Doublet-Lattic Method for Calculating 
%        Lift Distributions on Oscillating Surfaces in Subsonic Flows
% 
% ref 2: Katz & Plodkin, 'Low speed aerodynamics', second edition (for VLM
%        implementation)
%
% Inputs:
% 1. Panel = 5 column array defining [panel number, southwest node #, northwest node #, southeast node #, northeast node #]
%             the node number is the same number as defined in the first column of Node
%
% 2. Node = 4 column array defining [node #, x coordinate, y coordinate, z coordinate]
%
% 3. Mach = Mach number
%
% 4. k = k1 term from ref 1.: omega/U
%    omega = frequency of oscillation
%    U = freestream velocity
% 
% 5. S: panel areas
%
% 6. n_hat_w: normal vector part in vertical direction
% 7. n_hat_wl: normal vector part in lateral direction
% 
% n_hat_w and n_hat_wl indicate the direction of the normal vector for all
% panels. n_hat_w contains cos(gamma) where 'gamma' is the dihedral angle
% of a panel. n_hat_wl contains sin(gamma) of the panels. This information
% is useful in the VLM code
%
% Outputs
% AIC = aerodynamic influence coefficient matrix
%
% Code written by
% Aditya Kotikalpudi
% Graduate Assistant
% University of Minnesota
%
%
% Code based on:
% Original code by Frank R. Chavez, Iowa State University (~2002)
% Modified version by:
% Brian P. Danowsky
% P. Chase Schulze
% (c) Systems Technology, Inc. 2014
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% determine number of aero panels present
[N,m]=size(Panel);

% rows of D are the downwash locations while columns define the effects of
% each aero panels doublet line/horseshoe vortex on that rows' downwash location

% define downwash location (3/4 chord and half span of the aero panel)
P0 = zeros(N,3);
P0(:,1) = (Node(Panel(:,2),2)+Node(Panel(:,3),2) + ...
        3*(Node(Panel(:,4),2)+Node(Panel(:,5),2)))/8; %xcp
P0(:,2) = (Node(Panel(:,2),3)+Node(Panel(:,3),3))/2; %ycp
P0(:,3) = (Node(Panel(:,2),4)+Node(Panel(:,3),4))/2; %zcp

% define doublet locations (1/4 chord and 0, half span and full span of the
% aero panel), kernel is computed at that 3 points and a parabolic function
% is fitted to approximate the kernel along the doublet line (ref 1,
% equation 7).
P1 = zeros(N,3);
P1(:,1) = Node(Panel(:,2),2)+ ...
         (Node(Panel(:,4),2) - Node(Panel(:,2),2))/4; %xp1
P1(:,2) = Node(Panel(:,2),3); %yp1
P1(:,3) = Node(Panel(:,2),4); %zp1

% P3 is doublet point at tip of the panel
P3 = zeros(N,3);
P3(:,1) = Node(Panel(:,3),2)+ ...
         (Node(Panel(:,5),2) - Node(Panel(:,3),2))/4; %xp3
P3(:,2) = Node(Panel(:,3),3); %yp3
P3(:,3) = Node(Panel(:,3),4); %zp3

% P2 is doublet point at the half-span location of the panel
P2 = zeros(N,3);
P2(:,1) = (P1(:,1)+P3(:,1))/2;
P2(:,2) = (P1(:,2)+P3(:,2))/2;
P2(:,3) = (P1(:,3)+P3(:,3))/2;

% define half span length and chord at centerline for each panel
s = (0.5*((Node(Panel(:,3),3) - Node(Panel(:,2),3))' + ...
          (Node(Panel(:,3),4) - Node(Panel(:,2),4))'));
c = (     (Node(Panel(:,4),2) - Node(Panel(:,2),2) + ...
           Node(Panel(:,5),2) - Node(Panel(:,3),2))/2)';

% get the downwash effect from DLM and VLM implementations for the defined geometry 
Dd  = getDLM(P0,P1,P2,P3,s,c,k,Mach);           % DLM (oscillatory effects)
Dv  = getVLM(P0,P1,P3,S,Mach,n_hat_w,n_hat_wl); % VLM (steady state effects)
D   = Dv + Dd;
AIC = -inv(D);

end


function D = getDLM(P0,P1,P2,P3,e,cav,k,Mach)
%
% This function solves equation 6 using equation 7 from ref 1. Implements 
% the parabolic distribution assumption for the numerator of the
% incremental Kernel function. 
% 
%
% ref 1: Albano and Rodden - A Doublet-Lattic Method for Calculating 
%        Lift Distributions on Oscillating Surfaces in Subsonic Flows
%
% ref 2: Rodden,Taylor and McIntosh, 'Further refinement of Doublet lattice
%        method
% chord refers to chord of the aero panel
% span refers to the span of the aero panel
% 
% Input
%
% P0 = downwash recieving location x-y pair (1/2span, 3/4chord)
% P1 = root doublet location x-y pair (0 span, 1/4chord)
% P2 = semi-span doublet location x-y pair (1/2span, 1/4chord)
% P3 = tip doublet location x-y pair (1span, 1/4chord)
% e = half span length of the aero panel
% cav = centerline chord of the aero panel
% k = k1 term from ref 1.: omega/U
%   omega = frequency of oscillation
%   U = freestream velocity
% M = Mach number
%
% Output
%
% D = normalwash factor for the aero panel in question due to the doublets
% at the locations defined by P1, P2 and P3
%

N = size(P0,1);

%% Get relevant inputs for calculating Kernel function 
%(Page 1, reference 1,beginning with eq. 2)

% root doublet
% define the x,y distance from this doublet to the recieving point at 1/2span 3/4chord location
x01= bsxfun(@minus,P0(:,1),P1(:,1)');
y01= bsxfun(@minus,P0(:,2),P1(:,2)');
z01= bsxfun(@minus,P0(:,3),P1(:,3)');

% semi-span doublet
% define the x,y distance from this doublet to the recieving point at 1/2span 3/4chord location
x02= bsxfun(@minus,P0(:,1),P2(:,1)');
y02= bsxfun(@minus,P0(:,2),P2(:,2)');
z02= bsxfun(@minus,P0(:,3),P2(:,3)');

% tip doublet
% define the x,y distance from this doublet to the recieving point at 1/2span 3/4chord location
x03= bsxfun(@minus,P0(:,1),P3(:,1)');
y03= bsxfun(@minus,P0(:,2),P3(:,2)');
z03= bsxfun(@minus,P0(:,3),P3(:,3)');

%% cos and sin angles of all doublet line dihedrals
cosGamma = (P3(:,2) - P1(:,2))./(sqrt((P3(:,3)-P1(:,3)).^2 ...
         + (P3(:,2)-P1(:,2)).^2)); % y/sqrt(z^2 + y^2)
sinGamma = (P3(:,3) - P1(:,3))./(sqrt((P3(:,3)-P1(:,3)).^2 ...
         + (P3(:,2)-P1(:,2)).^2)); % z/sqrt(z^2 + y^2)

%% Kernel function (K) calculation
% Kappa (defined on page 3, reference 1) is calculated. The steady part of
% Kappa (i.e. reduced frequency = 0) is subtracted out and later
% compensated for by adding downwash effects from a VLM code. This ensures 
% that the doublet lattice code converges to VLM results under steady
% conditions. (Ref 2, page 3, equation 9)

% numerator of the singular kernel for this doublet
Ki_w = getKappa(x01,y01,z01,cosGamma,sinGamma,k,Mach);
Ki_0 = getKappa(x01,y01,z01,cosGamma,sinGamma,0,Mach);
Ki = Ki_w - Ki_0;       
                        
% numerator of the singular kernel for this doublet
Km_w = getKappa(x02,y02,z02,cosGamma,sinGamma,k,Mach);
Km_0 = getKappa(x02,y02,z02,cosGamma,sinGamma,0,Mach);
Km = Km_w - Km_0;

% numerator of the singular kernel for this doublet
K0_w = getKappa(x03,y03,z03,cosGamma,sinGamma,k,Mach);
K0_0 = getKappa(x03,y03,z03,cosGamma,sinGamma,0,Mach);
K0 = K0_w - K0_0;

%% Parabolic approximation of incremental Kernel function (ref 1, equation 7)
% define terms used in the parabolic approximation
e1 = abs(repmat(e,N,1));
A = (Ki-2*Km+K0)./(2*e1.^2);
B = (K0-Ki)./(2*e1);
C = Km;

% define r1,n0,zeta0
cosGamma = repmat(cosGamma',N,1);
sinGamma = repmat(sinGamma',N,1);
n0 = (y02.*cosGamma) + (z02.*sinGamma);
zeta0 = -(y02.*sinGamma) + (z02.*cosGamma);
r2 = sqrt((n0.^2) + (zeta0.^2));

% normalwash matrix factor I
I = (A.*(2*e1))+((0.5*B+n0.*A).*log((r2.^2 - 2*n0.*e1 + e1.^2)./(r2.^2 + 2*n0.*e1 + e1.^2)))...
    +(((n0.^2 - zeta0.^2).*A+n0.*B+C)./abs(zeta0).*atan(2*e1.*abs(zeta0)./(r2.^2 - e1.^2)));
% limit when zeta -> 0
ind = find(zeta0==0);
I2 = ((A.*(2*e1))+((0.5*B+n0.*A).*log(((n0-e1)./(n0+e1)).^2))...
    +((n0.^2).*A+n0.*B+C).*((2*e1)./(n0.^2 - e1.^2)));
I(ind) = I2(ind);

% normalwash matrix
D = repmat(cav,N,1).*I/(pi*8);

end


function Dfinal = getVLM(P0,P1,P3,PAreas,M,n_hat_w,n_hat_wl)
% code developed using Katz & Plodkin as reference. 
% ends the D matrix (downwash coeff) matrix, inverse of which gives the
% influence coeff matrix
% The AIC matrix as defined by Katz n plodkin provides vortex strength
% distribution across panels, not pressure difference. That factor needs to
% be accounted for, using the Hedman paper as reference
% multiply all y and z coordinates by beta = sqrt(1-M^2) to account for
% compressibility

%% multiply y coordinates with beta
% beta = sqrt(1-(M^2));
% P0(:,2) = beta*P0(:,2);
% P1(:,2) = beta*P1(:,2);
% P3(:,2) = beta*P3(:,2);
%% divide x coordinates with beta
% See Hedman 1965. 
% However, Hedman divides by beta^2 ... why?? 
beta = sqrt(1-(M^2));
P0(:,1) = P0(:,1)/beta;
P1(:,1) = P1(:,1)/beta;
P3(:,1) = P3(:,1)/beta;

%% get r1,r2,r0
epsilon = 10e-6;
% distance of vortex line beginning from control points
r1x= bsxfun(@minus,P0(:,1),P1(:,1)');
r1y= bsxfun(@minus,P0(:,2),P1(:,2)');
r1z= bsxfun(@minus,P0(:,3),P1(:,3)');

% distance of vortex line end from control points
r2x= bsxfun(@minus,P0(:,1),P3(:,1)');
r2y= bsxfun(@minus,P0(:,2),P3(:,2)');
r2z= bsxfun(@minus,P0(:,3),P3(:,3)');

% vortex line lengths
r0x = P3(:,1) - P1(:,1);
r0x = repmat(r0x,1,size(P0,1))';

r0y = P3(:,2) - P1(:,2);
r0y = repmat(r0y,1,size(P0,1))';

r0z = P3(:,3) - P1(:,3);
r0z = repmat(r0z,1,size(P0,1))';

% get normal vectors
n_hat_w = repmat(n_hat_w,1,size(P0,1));  % indicates cosine of panel dihedrals 
                                         % (1 for wing, 0 for winglet panels)
n_hat_wl = repmat(n_hat_wl,1,size(P0,1)); % sine of panel dihedrals 

%% induced velocity due to finite vortex line
r1Xr2_x = (r1y.*r2z) - (r2y.*r1z);
r1Xr2_y = -(r1x.*r2z) + (r2x.*r1z);
r1Xr2_z = (r1x.*r2y) - (r2x.*r1y);
mod_r1Xr2 = sqrt((r1Xr2_x.^2)+(r1Xr2_y.^2)+(r1Xr2_z.^2));

mod_r1 = sqrt((r1x.^2) + (r1y.^2) + (r1z.^2));
mod_r2 = sqrt((r2x.^2) + (r2y.^2) + (r2z.^2));

r0r1 = (r0x.*r1x) + (r0y.*r1y) + (r0z.*r1z);
r0r2 = (r0x.*r2x) + (r0y.*r2y) + (r0z.*r2z);

one = ones(size(r1Xr2_x));
D1_base = (1/(4*pi))*(one./(mod_r1Xr2.^2)).*((r0r1./mod_r1) - (r0r2./mod_r2));
D1_u = r1Xr2_x.*D1_base;
D1_v = r1Xr2_y.*D1_base;
D1_w = r1Xr2_z.*D1_base;

% adjust for non zero D matrix
ind =  find(mod_r1<epsilon);
D1_u(ind) = 0;
D1_v(ind) = 0;
D1_w(ind) = 0;

ind =  find(mod_r2<epsilon);
D1_u(ind) = 0;
D1_v(ind) = 0;
D1_w(ind) = 0;

ind =  find(mod_r1Xr2<epsilon);
D1_u(ind) = 0;
D1_v(ind) = 0;
D1_w(ind) = 0;

% get final D1 matrix 
% D1 matrix contains the perpendicular component of induced velocities at all panels.
% For wing panels, it's the z component of induced velocities (D1_w) while for
% winglets, it's the y component of induced velocities (D1_v)
D1 = (D1_w.*n_hat_w) + (D1_v.*n_hat_wl);

%% induced velocity due to inner semi-infinite vortex line

d2 = sqrt((r1y.^2) + (r1z.^2)); 
cosBB1 = 1;
cosBB2 = -r1x./mod_r1;
cosGamma = r1y./d2;
sinGamma = -r1z./d2;

D2_base = -(1/(4*pi))*(cosBB1 - cosBB2)./d2;
D2_u = zeros(size(D2_base));
D2_v = sinGamma.*D2_base;
D2_w = cosGamma.*D2_base;

ind = find(mod_r1<epsilon);
D2_u(ind) = 0;
D2_v(ind) = 0;
D2_w(ind) = 0;

% get final D2 matrix (same as D1)
D2 = (D2_w.*n_hat_w) + (D2_v.*n_hat_wl);

%% induced velocity due to outer semi-infinite vortex line
d3 = sqrt((r2y.^2) + (r2z.^2)); 
cosBT1 = r2x./mod_r2;
cosBT2 = -1;
cosGamma = -r2y./d3;
sinGamma = r2z./d3;

D3_base = -(1/(4*pi))*(cosBT1 - cosBT2)./d3;
D3_u = zeros(size(D3_base));
D3_v = sinGamma.*D3_base;
D3_w = cosGamma.*D3_base;

ind = find(mod_r2<epsilon);
D3_u(ind) = 0;
D3_v(ind) = 0;
D3_w(ind) = 0;

% get final D3 matrix (same as D1)
D3 = (D3_w.*n_hat_w) + (D3_v.*n_hat_wl);

%% total D
D = D1 + D2 + D3;

% Multiply additional factors to ensure that the D matrix maps pressure diff. 
% (rather than vortex strength) to downwash (ref Katz & Plodkin)  
% BUG found 05.02.2016, A. Voss
% deltaY = (P3(:,2) - P1(:,2)) + (P3(:,3) - P1(:,3));  % panel spans
deltaY = sqrt((P3(:,2) - P1(:,2)).^2 + (P3(:,3) - P1(:,3)).^2);  % panel spans
F = 0.5*PAreas./deltaY;
F = repmat(F,1,length(P0))';
Dfinal = D.*F;   

end


function kappa = getKappa(x0,y0,z0,cosGamma,sinGamma,k,M)
%% Function to calculate kappa
% this function calculates kappa as defined on page 3 of reference 1. The
% All the formulae are from page 1 of the reference.
% kappa = (r1^2) * K where K is the incremental Kernel function
% K = (K1T1 + K2T2)*exp(-jwx0/U)/(r1^2), where w is oscillation frequency
% variables passed to the function:

% x0 = x - chi (x is location of collocation pt (3/4th chord pt), chi is
%               location of doublet)
% y0 = y - eta
% z0 = z - zeta
% cosGamma: cosine of panel dihedral
% sinGamma: sine of panel dihedral
% k = w/U,  U is freestram velocity
% M: Mach no.

% Reference papers
% ref 1: Albano and Rodden - A Doublet-Lattic Method for Calculating 
%        Lift Distributions on Oscillating Surfaces in Subsonic Flows
%
% ref 2: Watkins, C. E., Hunyan, H. L., and Cunningham, H. J., "A Systematic 
%        Kernel Function Procedure for Determining Aerodynamic Forces on Oscillating 
%        or Steady Finite Wings at Subsonic Speeds," R-48, 1959, NASA.
%
% ref 3: Blair, Max. A Compilation of the mathematics leading to the doublet 
%        lattice method. No. WL-TR-92-3028. WRIGHT LAB WRIGHT-PATTERSON AFB OH, 1992.
%

%% declare all variables as defined in reference 1, page 1
%z0 = zeros(size(y0));
r1 = sqrt((y0.^2) + (z0.^2));
beta2 = (1-(M^2));
R = sqrt((x0.^2) + (beta2*(r1.^2)));
u1 = ((M*R) - x0)./(beta2*r1);
k1 = k*r1;
j = sqrt(-1);

%% calculate T1, T2
cos1 = repmat(cosGamma,1,length(cosGamma));
cos2 = repmat(cosGamma',length(cosGamma),1);
sin1 = repmat(sinGamma,1,length(sinGamma));
sin2 = repmat(sinGamma',length(sinGamma),1);

T1 = cos1.*cos2 + sin1.*sin2;
T2 = (z0.*cos1 - y0.*sin1).*(z0.*cos2 - y0.*sin2)./(r1.^2);

%% calculate I1 & I2
I1 = getI1(u1,k1);
I2 = getI2(u1,k1);

%% get kappa_temp
kappa_temp1 = I1 + ((M*r1).*exp(-j*(k1.*u1))./(R.*sqrt(1+(u1.^2))));
kappa_temp2 = -3*I2 - (j*k1.*(M.^2).*(r1.^2).*exp(-j*k1.*u1)./ ...
              ((R.^2).*sqrt(1+u1.^2))) - (M*r1.*((1+u1.^2).* ...
              ((beta2*r1.^2)./R.^2) + 2 + (M*r1.*u1./R))).* ...
              exp(-j.*k1.*u1)./(((1+u1.^2).^(3/2)).*R);
kappa_temp = kappa_temp1.*T1 + kappa_temp2.*T2;

%% Resolve the singularity arising when r1 = 0, ref 2, page 7, Eq 18
rInd = find(r1==0);  
kappa_temp(rInd(logical(x0(rInd)<0))) = 0;
kappa_temp(rInd(logical(x0(rInd)>=0))) = 2;
kappa = kappa_temp.*exp(-j*k*x0);

end


function I1 = getI1(u1,k1)
%% Function to get I1
% ref 1: Albano and Rodden - A Doublet-Lattic Method for Calculating 
%        Lift Distributions on Oscillating Surfaces in Subsonic Flows
%
% ref 3: Blair, Max. A Compilation of the mathematics leading to the doublet 
%        lattice method. No. WL-TR-92-3028. WRIGHT LAB WRIGHT-PATTERSON AFB OH, 1992.
%
% I1 described in eqn 3 of page1 of reference 1. Approximated as shown in
% page 3 of the reference. 
%

I1 = zeros(size(u1));
I1_0 = zeros(size(u1));
I1_neg = zeros(size(u1));
u_temp1 = zeros(size(u1));
u1_temp2 = zeros(size(u1));
k1_temp1 = zeros(size(u1));
k1_temp2 = zeros(size(u1));
%% evaluate I1 for u1>0
ind1 = find(u1>=0);
u_temp1(ind1) = u1(ind1);   % select elements in u1 > 0
k1_temp1(ind1) = k1(ind1);
I1_temp1 = getI1pos(u_temp1,k1_temp1);
I1(ind1) = I1_temp1(ind1);
j = sqrt(-1);

%% evaluate I1 for u1<0
% Method taken from ref 3, page 90, eq 275
ind2 = find(u1<0);
u1_temp2(ind2) = u1(ind2);
k1_temp2(ind2) = k1(ind2);

I1_0temp = getI1pos(zeros(size(u1)),k1_temp2);
I1_0(ind2) = I1_0temp(ind2);
I1_negtemp = getI1pos(-u1_temp2,k1_temp2);
I1_neg(ind2) = I1_negtemp(ind2);
I1(ind2) = (2*real(I1_0(ind2))) - real(I1_neg(ind2)) + (j*imag(I1_neg(ind2)));

end


function I1pos = getI1pos(u1,k1)
%% Function to get I1 for positive u1 values
% ref 1: Albano and Rodden - A Doublet-Lattic Method for Calculating 
%        Lift Distributions on Oscillating Surfaces in Subsonic Flows

% I1 described in eqn 3 of page1 of reference 1. Approximated as shown in
% page 3 of the reference. This implementation is only valid for u1>0. For
% u1<0, this function is still used to obtain I1 in an indirect manner as
% described in getI1.m, in which this function is called.
%

%% constants used in approximation

a1 = 0.101;
a2 = 0.899;
a3 = 0.09480933;
b1 = 0.329;
b2 = 1.4067;
b3 = 2.90;

%% solve for I1
j = sqrt(-1);
i1 = (a1*exp((-b1-(j*k1)).*u1)./(b1+(j*k1))) + ...
     (a2*exp((-b2-(j*k1)).*u1)./(b2+(j*k1)));
i2 = (a3./(((b3 + (j*k1)).^2) + (pi^2))).*(((b3+(j*k1)).*sin(pi.*u1)) + ...
     (pi*cos(pi.*u1))).*exp((-b3-(j*k1)).*u1);
I1_temp = i1 + i2;
I1pos = ((1-(u1./sqrt(1+u1.^2))).*exp(-j*k1.*u1)) - (j*k1.*I1_temp);

end          


function I2 = getI2(u1,k1)
%% Function to get I2
% ref 1: Albano and Rodden - A Doublet-Lattic Method for Calculating 
%        Lift Distributions on Oscillating Surfaces in Subsonic Flows
%
%ref 3: Blair, Max. A Compilation of the mathematics leading to the doublet 
%       lattice method. No. WL-TR-92-3028. WRIGHT LAB WRIGHT-PATTERSON AFB OH, 1992.
%
% I2 described in eqn 3 of page1 of reference 1. Approximated as mentioned on
% page 3 of the reference. 
%

I2 = zeros(size(u1));
I2_0 = zeros(size(u1));
I2_neg = zeros(size(u1));
u_temp1 = zeros(size(u1));
u1_temp2 = zeros(size(u1));
k1_temp1 = zeros(size(u1));
k1_temp2 = zeros(size(u1));

%% calculate I2
ind1 = find(u1>=0);
u_temp1(ind1) = u1(ind1);   % select elements in u1 > 0
k1_temp1(ind1) = k1(ind1);
I2_temp1 = getI2pos(u_temp1,k1_temp1);
I2(ind1) = I2_temp1(ind1);
j = sqrt(-1);
% Calulate integral I2(ref 1, page 1, eq 3) for u1<0
% Method taken from ref 3, page 90, eq 275
ind2 = find(u1<0);
u1_temp2(ind2) = u1(ind2);
k1_temp2(ind2) = k1(ind2);

I2_0temp = getI2pos(zeros(size(u1)),k1_temp2);
I2_0(ind2) = I2_0temp(ind2);
I2_negtemp = getI2pos(-u1_temp2,k1_temp2);
I2_neg(ind2) = I2_negtemp(ind2);
I2(ind2) = (2*real(I2_0(ind2))) - real(I2_neg(ind2)) + (j*imag(I2_neg(ind2))); 

end
     
         
function I2pos = getI2pos(u1,k1)
% this function gets I2 integral for non-planar body solutions, for u1>0
% I2 = I2_1 + I2_2
% Expressions for I2_1 & I2_2 have been derived using the same
% approximations as those for I1 
a1 = 0.101;
a2 = 0.899;
a3 = 0.09480933;
b1 = 0.329;
b2 = 1.4067;
b3 = 2.90;
i = sqrt(-1);
eiku = exp(-i.*k1.*u1);

I2_1 = getI1pos(u1,k1);
I2_2_1 = a1.*exp(-(b1+i*k1).*u1)./((b1+i*k1).^2) + ...
         a2.*exp(-(b2+i*k1).*u1)./((b2+i*k1).^2) + ...
         ((a3.*exp(-(b3+i*k1).*u1)./(((b3+i*k1).^2 + ...
         pi^2).^2)).*(pi*((pi*sin(pi*u1)) - ((b3+i*k1).*cos(pi*u1))) - ...
         ((b3+i*k1).*(pi*cos(pi*u1) + ((b3+i*k1).*sin(pi*u1))))));  
I2_2 = (eiku.*(u1.^3)./((1+u1.^2).^(3/2)) - getI1pos(u1,k1) - ...
       eiku.*u1./sqrt((1+u1.^2)))/3 - (k1.*k1.*I2_2_1/3);
  
I2pos = I2_1 + I2_2;

end          


