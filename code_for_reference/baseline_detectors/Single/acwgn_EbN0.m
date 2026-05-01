function [RxSig, Nv, noise]= acwgn_EbN0(TxSig, Es, SNR, Nt, Nr, Mc, FECRate)
% AWGN channel for MIMO/SISO channel, using this equation: Eb/N0 = (Es*Nr*Nt)/(Nt*Mc*FECRate*N0),
% where Es is the transmit symbol energy, Nr is the receive antenna number, Nt is the
% the transmint antenna number, Mc is bit/symbol for different constelation of modulation, Rate is the
% FEC rate or equivelant rate, for example spreading factor. 
%
%	function [RxSig, sigma2] = acwgn_EbN0(TxSig, Es, SNR, Nt, Nr, Mc, FECRate)
%
%	Parameters:
%
%	Input:
%			TxSig: 		The transmit signal, 2-D, column as time direction, row as antenna direction at the receiver.
%			Es:			Transmit symbol energy.
%			SNR:		Signal to noise rate, Eb/N0, dB;
%			Nt:			Transmit antenna number.
%			Nr:			Receive antenna number.
%			Mc:			Bit/Symbol for different constelation
%			FECRate:	Forward error correcting code rate, or equivelant rate, for example spreading factor
%	Equation:
%			Eb       Es*M*N
%			-- =  ------------
%			N0      N0*R*Mc*M
%
%	Output:
%			RxSig:		Received signal after adding the noise.
%			sigma2:		Noise variance at the receiver.
%
% See also:
%
% (C) Wang Dongming, NCRL, SEU, MAR. 2003

TxE = Es*Nr*Nt;

% noise variance
sigma2 = TxE/(10^(SNR/10)*Mc*FECRate*Nt);

% output of the AWGN channel
noise = sqrt(sigma2/2)*(randn(size(TxSig))+1i*randn(size(TxSig)));
RxSig = TxSig + noise ;

Nv = sigma2;
