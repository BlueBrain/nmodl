: simple first-order model of calcium dynamics

        NEURON {
        SUFFIX ican
        USEION n READ ni, in WRITE ni
        RANGE n
        GLOBAL depth,ninf,taur
        RANGE var

        }

        UNITS {
        (molar) = (1/liter)
        (mM) = (milli/liter)
        (um)	= (micron)
        (mA) = (milliamp)
        (msM)	= (ms mM)
        FARADAY    = (faraday) (coul)
        }

        PARAMETER {
        depth	= .1	(um)
        taur =  200 (ms)	: rate of calcium removal for stress conditions
        ninf	= 50e-6(mM)	:changed oct2
        ni		(mM)
        }

        ASSIGNED {
        in		(mA/cm2)
        drive_channel	(mM/ms)
        var     (mV)
        }

        STATE {
        n		(mM)
        }


        BREAKPOINT {
        SOLVE state METHOD euler
        }

        INCLUDE "var_init.inc"

        DERIVATIVE state {

        drive_channel =  - (10000) * in / (2 * FARADAY * depth)
        if (drive_channel <= 0.) { drive_channel = 0.  }   : cannot pump inward
        n' = drive_channel/18 + (ninf -n)/taur*11
	ni = n
}

INITIAL {
    var_init(var)
    n = ninf
}
