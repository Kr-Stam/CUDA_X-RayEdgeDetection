#pragma once

namespace gpu
{
	void bilateral_filter(unsigned char *src, unsigned char *gray, unsigned char *dest, int w, int h, int ww, int wh, double sigmaS, double sigmaB);
	void gauss_filter(unsigned char *src, unsigned char *gray, unsigned char *dest, int w, int h, double sigmaS, int mw, int mh);
}
