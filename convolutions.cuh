#pragma once

namespace gpu {
	void conv(const unsigned char *src, unsigned char *dest, int w, int h, const double *mask_t, int mw, int mh);
	void conv_constant(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const double *mask_t, int mw, int mh);
	void conv_tiled(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const double *mask_t, int mw, int mh);

	void conv_3ch_2d(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const double *mask_t, int mw, int mh);
	void conv_3ch_2d_constant(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const double *mask_t, int mw, int mh);
	void conv_3ch_tiled(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const double *mask_t, int mw, int mh);
	void conv_range(const unsigned char* src_h, unsigned char* dest_h, int w, int h, int mw, int mh);
}
