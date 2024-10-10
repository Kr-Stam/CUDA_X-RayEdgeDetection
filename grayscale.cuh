#pragma once

namespace gpu
{
	void grayscale_avg(const unsigned char *src_h, unsigned char *dest_h, int h, int w);
	void grayscale_avg_3ch_1ch(const unsigned char *src_h, unsigned char *dest_h, int h, int w);
}
