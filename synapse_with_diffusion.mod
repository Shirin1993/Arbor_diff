NEURON {
	POINT_PROCESS synapse_with_diffusion
	USEION s WRITE sd
	RANGE sV
}

PARAMETER {	area diam }
ASSIGNED { volume sV }
STATE { sVs }

INITIAL {
	volume = 0.25 * area * diam
	sV = sd * volume
	sVs = sd * volume
}

BREAKPOINT {
	sV = sd * volume
	sVs = sd * volume
}

NET_RECEIVE(weight) {
	sd = sd + 0.001*weight*area/volume
}
