INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX ican
	USEION n READ en WRITE in VALENCE 1
	USEION ca READ cai
        RANGE gbar, m_inf, tau_m, in
	GLOBAL beta, cac, taumin
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) = (millimolar)
}


PARAMETER {
	v		(mV)
	celsius	= 36	(degC)
	en	= -30	(mV)		: reversal potential
	cai 	= 2.4e-5 (mM)		: initial [Ca]i
	gbar	= 0.0004 (mho/cm2)
	beta	= 0.001	(1/ms)		: backward rate constant
	cac	= 0.013	(mM)		: middle point of activation fct
	taumin	= 0.11	(ms)		: minimal value of time constant
}


STATE {
	m
}

ASSIGNED {
	in	(mA/cm2)
	m_inf
	tau_m	(ms)
	tadj
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	in = gbar * m*m * (v - en)
}






