# Deprecated compatibility alias

`su2017_hessenberg` existed in the original repository but did **not** faithfully implement a paper: it embedded QIM into a selected Hessenberg-`H` coefficient.

It now delegates to the audited Su (2016) Hessenberg implementation, whose paper modifies `q22/q32` in the orthogonal matrix `Q` with `T=0.042`.

Use `su2016_hessenberg` in new experiment configurations. The alias is retained only so old commands do not break.
