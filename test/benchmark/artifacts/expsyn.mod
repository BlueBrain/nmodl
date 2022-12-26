NEURON {
	POINT_PROCESS ExpSyn
	RANGE tau, e, i
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau = 0.1 (ms) <1e-9,1e9>
	e = 0	(mV)
}

ASSIGNED {
	v (mV)
	i (nA)
}

STATE {
	g_state (uS)
}

INITIAL {
	g_state=0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = g_state*(v - e)
}

DERIVATIVE state {
	g_state' = -g_state/tau
}

NET_RECEIVE(weight (uS)) {
	g_state = g_state + weight
}
