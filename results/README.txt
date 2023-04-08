the first collection is for lam = 0.01 : 0.4 ;  dil 0 : 0.5   ;   gamma = 0.15
the second is  lam = 0.01 : 0.5 ;  dil 0 : 1.0  ;  gamma = 0.15 ; d = 3; N = 10 000, iter = 30
the third is  lam = 0.01 : 0.8 ;  dil 0 : 1.0  ;  gamma = 0.03 ; d = 3; N = 10 000, iter = 30  
USED

the fourth collection is mismatched with lam_p = 0.3 and lam_i = 0.01 : 0.8; dil=0:0.2, gam = 0.03, d=3, N=40000, iter = 60
the fifth collection is optimal lam = 0.01 : 0.5 ;  dil 0 : 1.0  ;  gamma = 0.03 ; d = 5; N = 10 000, iter = 30
the sixth collection is mismatched with lam_p = 0.3 and lam_i = 0.01 : 0.4; dil=0:1, gam = 0.03, d=3, N=40000, iter = 60
the 7th collection is mismatched with lam_p = 0.3 and lam_i = 0.01 : 0.4; dil=0:0.2, gam = 0.03, d=3, N=100000, iter = 60
the 8th collection is mismatched with lam_p = 0.3 and lam_i = 0.01 : 0.8; dil=0:1.0, gam = 0.03, d=3, N=40000, iter = 60
USED

the 9th collection is mismatched in gamma with lam = 0.3 and gam_i = 0.001 : 0.4; dil=0:1.0, gam = 0.05, d=3, N=40000, iter = 60
the 10th collection is mismatched WITH LEARNING in gamma and lam with lamp = gamp= 0.05:0.9 and gam_i = 0.1 = lam_i (initial condition); dil=0.0, d=3, N=10000
the 11th collection is mismatched WITH LEARNING in gamma and lam with lamp = gamp= 0.05:0.9 and gam_i = 0.5 = lam_i (initial condition); dil=0.0, d=3, N=10000
USED

the 12th collection is mismatched WITH LEARNING in lam with lamp = 0.05:0.9 and dil = 0:1 lam_i = 0.1 (initial condition); gam_p = gam_i = 0.1, d=3, N=20000
USED

the 13th collection is mismatched WITH LEARNING in lam with lamp = 0.05:0.9 and dil = 0.1:1.0 lam_i = 0.5 (initial condition); gam_p = gam_i = 0.1, d=3, N=20000, AUC on NON observed


the 14th collection is mismatched with lam_p = 0.3 and lam_i = 0.01 : 0.8; dil=0.1:1.0, gam = 0.03, d=3, N=40000, iter = 500, AUC non observed

the 15th collection is scattered 3:T T=8 with lam_p = 0.05:0.5, gam_p = 0.05:0.5, N=20000, iter = 5+20, dil=9/10, fr = 0, AUC non observed

the 16th collection is scattered 3:T T=15 with lam_p = 0.05:0.5, gam_p = 0.05:0.5, N=20000, iter = 5+20, dil=9/10, fr = 0, AUC non observed

the 17th collection is mismatched WITH LEARNING in gamma and lam with lamp = gamp= 0.05:0.9 and gam_i = 0.5 = lam_i (initial condition); dil=0.5, d=3, N=20000 iters=60

the 18th is optimal with lam = 0.05 : 0.9 ;  dil 0.1 : 1.0  ;  gamma = 0.1 ; Poisson(3); N = 20 000, iter = 100,  AUC non observed; look at 13th

the 19th is optimal with lam = 0.05 : 0.9 ;  dil 0.1 : 1.0  ;  gamma = 0.1 ; Poisson(3); N = 20 000, iter = 100,  AUC observed; look at 13th

the 20th collection is optimal in gamma and lam with lamp = gamp= 0.05:0.9  dil=0.5, d=3, N=20000 iters=60 AUC obs

the 21th collection is the 17th but with NaN in the AUC

the 22th collection is optimal in gamma and lam with lamp = gamp= 0.05:0.9  dil=0.5, d=3, N=40000 iters=100 AUC all.

-20 th and 17-21 th correspond "comparison_opt_vs_learn/"
- 18 and 19 correspond
-18th and 19th correspond and are similar to 3rd
-15th and 16th collections are correspondent
-14th is the iterations mega result, in case we still have 11th

