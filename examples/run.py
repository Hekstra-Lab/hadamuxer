from pylab import *
import fabio

cbfs = [
    "images/lysozyme_EFX53_1_0000001.cbf",
    "images/lysozyme_EFX53_1_0000002.cbf",
    "images/lysozyme_EFX53_1_0000003.cbf",
    "images/lysozyme_EFX53_1_0000004.cbf",
    "images/lysozyme_EFX53_1_0000005.cbf",
    "images/lysozyme_EFX53_1_0000006.cbf",
    "images/lysozyme_EFX53_1_0000007.cbf",
    "images/lysozyme_EFX53_1_0000008.cbf",
    "images/lysozyme_EFX53_1_0000009.cbf",
    "images/lysozyme_EFX53_1_0000010.cbf",
    "images/lysozyme_EFX53_1_0000011.cbf",
]

pixels = [fabio.open(cbf).data for cbf in cbfs]
pixels = [np.where((im > 4_000) | (im < 0), 0., im) for im in pixels]

from hadamuxer.solver import Solver

s = Solver(pixels)
s = s.cuda()
s.solve()

result = s.time_points


from IPython import embed
embed(colors='linux')
