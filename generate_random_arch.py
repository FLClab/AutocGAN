import numpy

CONV_TYPE = {0: "post", 1: "pre"}
NORM_TYPE = {0: None, 1: "bn", 2: "in"}
UP_TYPE = {0: "bilinear", 1: "nearest", 2: "deconv"}
SHORT_CUT_TYPE = {0: False, 1: True}
SKIP_TYPE = {0: False, 1: True}

for _ in range(10):
	arch = [
		numpy.random.randint(0,2),
		numpy.random.randint(0,3),
		numpy.random.randint(0,3),
		numpy.random.randint(0,2),
		numpy.random.randint(0,2),
		numpy.random.randint(0,3),
		numpy.random.randint(0,3),
		numpy.random.randint(0,2),
		numpy.random.randint(0,2),
		numpy.random.randint(0,2),
		numpy.random.randint(0,3),
		numpy.random.randint(0,3),
		numpy.random.randint(0,2),
		numpy.random.randint(0,4)]

	print('{} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(arch[0], arch[1], arch[2], arch[3], arch[4], arch[5], arch[6], arch[7], arch[8], arch[9], arch[10], arch[11], arch[12], arch[13]))