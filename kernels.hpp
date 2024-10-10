#pragma once

//------------edge-detecting kernels---------------
extern const double Dx_1d[3];
extern const double Dx_2x2[9];
extern const double Dx_diagonal_2x2[9];
extern const double Dx_3x3[9];
extern const double Dx_3x3_t[9];
extern const double Dx_5x5[25];

extern const double Dy_2x2[9];
extern const double Dy_diagonal_2x2[9];
extern const double Dy_3x3[9];

extern const double Dt_3x3[9];
extern const double Dt_3x3_n[9];

extern const double Dz_2x2[9];

//---------------filter-kernels--------------------
extern const double GAUS_KERNEL_5x5[25];
extern const double GAUS_KERNEL_3x3[9];
