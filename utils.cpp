#include "utils.hpp"

void utils::upscale_3ch(unsigned char *src, int w, int h, int scale_factor, unsigned char *dest)
{
		int offset = 1 << scale_factor;
		for (int i = 0; i < h; i++)
		{
				for (int j = 0; j < w; j++)
				{
						int pos = i * w + j;
						for (int p = 0; p < offset; p++)
						{
								for (int q = 0; q < offset; q++)
								{
										int tmp_pos = (i * offset + p) * (w * offset) + (j * offset) + q;
										dest[tmp_pos * 3 + 0] = src[pos * 3 + 0];
										dest[tmp_pos * 3 + 1] = src[pos * 3 + 1];
										dest[tmp_pos * 3 + 2] = src[pos * 3 + 2];
								}
						}
				}
		}
}

void utils::upscale_1ch(unsigned char *src, int w, int h, int scale_factor, unsigned char *dest)
{
		int offset = 1 << scale_factor;
		for (int i = 0; i < h; i++)
		{
				for (int j = 0; j < w; j++)
				{
						int pos = i * w + j;
						for (int p = 0; p < offset; p++)
						{
								for (int q = 0; q < offset; q++)
								{
										int tmp_pos = (i * offset + p) * (w * offset) + (j * offset) + q;
										dest[tmp_pos] = src[pos];
								}
						}
				}
		}
}

int utils::generate_gaussian_kernel(double sigmaS, int kernel_size, double *dest)
{
		if (kernel_size == -1)
				kernel_size = (int) (2.0 * M_PI * sigmaS);
		if (kernel_size % 2 == 0)
				kernel_size += 1;

		if(dest == nullptr)
			return 1;
		
		double *gaus_mask = dest;
		int hk = kernel_size >> 1;

		for (int i = 0; i < hk + 1; i++)
		{
				for (int j = 0; j < hk + 1; j++)
				{
						double sigmaS2 = sigmaS * sigmaS;

						double m = i;
						double n = j;
						double n2 = n * n;
						double m2 = m * m;
						double value = 1.0 / (2.0 * M_PI * sigmaS2) * pow(M_E, -0.5 * (n2 + m2) / sigmaS2);

						gaus_mask[(hk + i) * kernel_size + hk + j] = value;
						gaus_mask[(hk - i) * kernel_size + hk - j] = value;
						gaus_mask[(hk + i) * kernel_size + hk - j] = value;
						gaus_mask[(hk - i) * kernel_size + hk + j] = value;
				}
		}
		
		//Normalize the values in the range [0, 1]
		double sum = 0;
		for (int i = 0; i < kernel_size; i++)
		{
				for (int j = 0; j < kernel_size; j++)
				{
						sum += gaus_mask[i * kernel_size + j];
				}
		}
		for (int i = 0; i < kernel_size; i++)
		{
				for (int j = 0; j < kernel_size; j++)
				{
						gaus_mask[i * kernel_size + j] /= sum;
				}
		}

		return 0;
}
