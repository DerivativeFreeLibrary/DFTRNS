clear all
close all

x_initial = ones(10,1);
fhandle   = @l1hilbert;
omega     = 1;
MAXNF     = 10000;
iprint    = 1;
[f,x]     = Advanced_DFO_TRNS(fhandle,x_initial,omega,MAXNF,iprint);
