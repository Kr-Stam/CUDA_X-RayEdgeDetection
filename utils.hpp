#include <math.h>

namespace utils
{
	
	template<typename T> inline void clamp(T& num, T low, T high)
	{
		if(num < low)
			num = low;
		else if (num > high)
			num = high;
	}

	void upscale_3ch(unsigned char *src, int w, int h, int scale_factor, unsigned char *dest);

	void upscale_1ch(unsigned char *src, int w, int h, int scale_factor, unsigned char *dest);

	/**
	 * \brief Generate a gaussian kernel mask
	 * \param sigmaS      : standard deviation of the gaussian, if not specified then 1 by default
	 * \param kernel_size : desired kernel size, if not specified or -1 then it is the optimal kernel size for the value of sigma
	 * \param dest        : a destination array of size (kernel_size x kernel_size)
	 *
	 * \note If the kernel size is even it is automatically made odd, as it is more applicable for convolution masks
	 * \note The destination kernel MUST be allocated
	 *
	 * \return 0 if succesful, 1 if dest was not passed in
	 */
	int generate_gaussian_kernel(double sigmaS = 1, int kernel_size = -1, double *dest = nullptr);

}
