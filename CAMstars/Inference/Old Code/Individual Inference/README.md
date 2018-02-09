In this directory we use Nested Sampling to infer abundances for individual stars.
For each star we infer the bulk and accreting abundance of each element, using
tight priors on each and a relation between the two based on the accretion rate and
refractory enhancement/depletion.

Two versions of the code are available.
One, infer_f, assumes nothing about the accreted abundance in the photosphere,
while infer_star uses data about the star and the accretion rate to infer this
abundance.
