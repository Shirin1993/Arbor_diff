NEURON {
	SUFFIX neuron_with_diffusion
	USEION s READ sd
	RANGE sV
}

PARAMETER {	area diam }
ASSIGNED { volume sV }
STATE { sVs }

INITIAL {
	volume = 0.25 * diam * area
	sV = sd * volume
	sVs = sd * volume
}

BREAKPOINT {
	sV = sd * volume
	sVs = sd * volume
}
