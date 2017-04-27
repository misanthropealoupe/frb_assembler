#include <fftw3.h>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <png++/png.hpp>

#ifndef RF_HEADER
#define RF_HEADER
#include "rf_pipelines.hpp"
#include "rf_pipelines_internals.hpp"
#endif

#include "mpl_interface.hpp"

using namespace std;
using namespace rf_pipelines;

const float pi = (atan(1)*4.0);

float blackman_fun(int index, int width){
	float coef = static_cast<float>(index)/static_cast<float>(width);
	return 0.42 - 0.5*cos(2.0*pi*coef) + 0.08*cos(4*pi*coef);
}

void neg_normalize(float* ar, ssize_t len){
	float sumf = 0;
	for(ssize_t i = 0; i < len; i++){
		sumf += ar[i];
	}
	for(ssize_t i = 0; i < len; i++){
		ar[i] = -ar[i]/sumf;
	}
}

void mult_inplace(float* a, float* b, ssize_t len){
	for(ssize_t i = 0; i < len; i++){
		a[i] = a[i]*b[i];
	}
}

void sub_inplace(float* a, float* b, ssize_t len){
	for(ssize_t i = 0; i < len; i++){
		a[i] -= b[i];
	}
}

void sqrt_inplace(float* a, ssize_t len){
	for(ssize_t i = 0; i < len; i++){
		a[i] = sqrt(a[i]);
	}
}

void add_inplace(float* a, float* b, ssize_t len){
	for(ssize_t i = 0; i < len; i++){
		a[i] += b[i];
	}
}

void div_inplace(float* a, float* b, ssize_t len){
	for(ssize_t i = 0; i < len; i++){
		a[i] /= b[i];
	}
}

float sum(float* ar, ssize_t len){
	float sumf = 0;
	for(ssize_t i = 0; i < len; i++){
		sumf += ar[i];
	}
	return sumf;
}

float mean(float* ar, ssize_t len){
	return sum(ar,len)/((float) len);
}

float* broadcast(float val, ssize_t len){
	float* ret = new float[len];
	for(ssize_t i = 0; i < len; i++){
		ret[i] = val;
	}
	return ret;
}

float* mult(float* a, float* b, ssize_t len){
	float* ret = new float[len];
	for(ssize_t i = 0; i < len; i++){
		ret[i] = a[i]*b[i];
	}
	return ret;
}

float* add(float* a, float* b, ssize_t len){
	float* ret = new float[len];
	for(ssize_t i = 0; i < len; i++){
		ret[i] = a[i] + b[i];
	}
	return ret;
}

float sample_var(float* ar, ssize_t len){
	float flen = (float) len;
	float* interim_sum = mult(ar,ar,len);
	float ret = ((flen - 1)/flen)*sum(interim_sum,len);
	delete[] interim_sum;
	return ret;
}

float skew(float*ar, ssize_t len){
	float flen = (float) len;
	float* inter_sum = mult(ar,ar,len);
	mult_inplace(inter_sum,ar,len);
	float ret = ((flen - 1)/flen)*sum(inter_sum,len);
	delete[] inter_sum;
	return ret;
}


float median(float* in, ssize_t len){
	vector<float> t_vec(in, in + len);
	sort(t_vec.begin(),t_vec.end());
	if(len % 2 == 0){
		ssize_t mid = len/2;
		return 0.5*(t_vec[mid-1] + t_vec[mid]);
	}
	else{
		return t_vec[(len - 1)/2];
	}
}

void mult_inplace_complex(fftw_complex* a, fftw_complex* b, ssize_t len){
	for(ssize_t i = 0; i < len; i++){
		a[i][0] = a[i][0]*b[i][0] - a[i][1]*b[i][1];
		a[i][1] = a[i][0]*b[i][1] + a[i][1]*b[i][0];
	}
}

void do_stdskew_mask(ssize_t nf, ssize_t nt, float sigma_cut, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride){
	float* vars = new float[nf];
	float* skews = new float[nf];
	#pragma omp parallel for
	for(ssize_t i = 0; i < nf; i++){
		float* a = mult(&intensity[i*stride],&weights[i],nt);
		float* b = mult(&intensity[i*stride],&weights[i],nt);
		vars[i] = sample_var(a,nt);
		skews[i] = skew(b,nt);
		delete[] a;
		delete[] b;
	}


	// uint8_t* chan_mask = new uint8_t[nf];
	// uint8_t* chan_mask_skew = new  uint8_t[nf];
	float meanvar =  mean(vars,nf);
	float stdvar = sqrt(sample_var(vars,nf));
	float meanskew =  mean(skews,nf);
	float stdskew = sqrt(sample_var(skews,nf));
	int n_rep = 0;
	#pragma omp parallel for
	for(ssize_t i = 0; i < nf; i++){
		if(vars[i] > sigma_cut*stdvar + meanvar){
			weights[i] = 0;
			for(int j = 0; j < nt; j++){
				intensity[i * stride + j] = 0.0;
			}
			n_rep++;
		}

		if(skews[i] > sigma_cut*stdskew + meanskew){
			weights[i] = 0;
			for(int j = 0; j < nt; j++){
				intensity[i * stride + j] = 0.0;
			}
			n_rep++;
		}
	}
	delete[] vars;
	delete[] skews;
	cout << "replaced " << n_rep << " channels!" << endl;
}


struct remove_subband : public wi_transform{
	std::vector<std::pair<int,int>> bands;
	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	ssize_t nt_maxwrite;
	//ssize_t nt_chunk;
	remove_subband(std::vector<std::pair<int,int>> bands);
	~remove_subband(){}
	virtual void set_stream(const wi_stream &stream);
	virtual void start_substream(int isubstream, double t0){}
	virtual void end_substream(){}
	virtual void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	std::string get_name() const{
		return std::string("remove_subband");
	}
};

remove_subband::remove_subband(std::vector<std::pair<int,int>> bands)
	: bands(bands)
{
	this->name = this->get_name();
}

void remove_subband::set_stream(const wi_stream &stream){
	this->nfreq = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;
}

void remove_subband::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride){
	for(const std::pair<int,int> &p : bands){
		std::cout << "zeroing band from index " << p.first << " to " << p.second << std::endl;
		for(int i = p.first; i < p.second; i++){
			for(int j = 0; j < nt_chunk; j++){
				intensity[i * stride + j] = 0.0;
			}
		}
	}
}


struct remove_noisy_freq : public wi_transform{
	float sigma_cut;

	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	ssize_t nt_maxwrite;
	//ssize_t nt_chunk;
	remove_noisy_freq(const float sigma_cut);
	~remove_noisy_freq(){}
	virtual void set_stream(const wi_stream &stream);
	virtual void start_substream(int isubstream, double t0){}
	virtual void end_substream(){}
	virtual void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	std::string get_name() const{
		return std::string("remove_noisy_freq");
	}
};

remove_noisy_freq::remove_noisy_freq(const float my_sigma_cut){
	this->name = this->get_name();
	sigma_cut = my_sigma_cut;
}

void remove_noisy_freq::set_stream(const wi_stream &stream){
	this->nfreq = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;
}

void remove_noisy_freq::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride){
	for(ssize_t i = 0; i < 2; i++){
		do_stdskew_mask(this->nfreq,this->nt_chunk,sigma_cut,intensity,weights,stride,pp_intensity,pp_weight,pp_stride);
		//cout << "ok" << endl;
	}
}

struct remove_continuum : public wi_transform{
	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	ssize_t nt_maxwrite;
	//ssize_t nt_chunk;
	remove_continuum(){
		this->name = this->get_name();
	}
	virtual void set_stream(const wi_stream &stream);
	virtual void start_substream(int isubstream, double t0){}
	virtual void end_substream(){}
	virtual void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	virtual std::string get_name() const{
		return std::string("remove_continuum");
	}
};

void remove_continuum::set_stream(const wi_stream &stream){
	this->nfreq = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;
}

void remove_continuum::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride){
	//mean-sub and 
	//sum continuum
	ssize_t nf = this->nfreq;
	ssize_t nt = this->nt_chunk;
	float* freq_sum = new float[nt] ();
	for(ssize_t i = 0; i < nf; i++){
		float ch_mean = mean(&intensity[i*stride],nt);
		float* bc = broadcast(ch_mean,nt);
		sub_inplace(&intensity[i*stride],bc,nt);
		add_inplace(freq_sum,&intensity[i*stride],nt);
		delete[] bc;
	}

	float* mul = mult(freq_sum,freq_sum,nt);
	float* bc = broadcast(sqrt(sum(mul,nt)),nt);
	div_inplace(freq_sum,bc,nt);
	delete[] mul;
	delete[] bc;

	#pragma omp parallel for
	for(ssize_t i = 0; i < nf; i++){
		float* mul = mult(&intensity[i*stride],freq_sum,nt);
		float* bc = broadcast(sum(mul,nt),nt);
		float* mul2 = mult(bc,freq_sum,nt);
		sub_inplace(&intensity[i*stride],mul2,nt);
		delete[] mul;
		delete[] bc;
		delete[] mul2;
	}
	delete[] freq_sum;
}

struct remove_outliers : public wi_transform{
	float sigma_cut;

	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	ssize_t nt_maxwrite;
	//ssize_t nt_chunk;
	remove_outliers(const float my_sigma_cut);
	virtual void set_stream(const wi_stream &stream);
	virtual void start_substream(int isubstream, double t0){}
	virtual void end_substream(){}
	virtual void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	virtual std::string get_name() const{
		return std::string("remove_outliers");
	}
};

remove_outliers::remove_outliers(const float my_sigma_cut){
	this->name = this->get_name();
	sigma_cut = my_sigma_cut;
}

void remove_outliers::set_stream(const wi_stream &stream){
	this->nfreq = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;
}

void remove_outliers::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride){
	ssize_t nf = this->nfreq;
	ssize_t nt = this->nt_chunk;
	#pragma omp parallel for
	for(ssize_t i = 0; i < nf; i++){
		float ch_mean = mean(&intensity[i*stride],nt);
		float ch_std = sqrt(sample_var(&intensity[i*stride],nt));
		for(ssize_t j = 0; j < nt; j++){
			if(abs(intensity[i*stride + j] - ch_mean) > sigma_cut*ch_std){
				intensity[i*stride + j] = ch_mean;
			}
		}
	}
	//fin
}

struct sys_temperature_bandpass : public wi_transform{
	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	ssize_t nt_maxwrite;
	//ssize_t nt_chunk;
	sys_temperature_bandpass();
	virtual void set_stream(const wi_stream &stream);
	virtual void start_substream(int isubstream, double t0){}
	virtual void end_substream(){}
	virtual void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	virtual std::string get_name() const{
		return std::string("sys_temperature_bandpass");
	}
};

sys_temperature_bandpass::sys_temperature_bandpass(){
	this->name = this->get_name();
}

void sys_temperature_bandpass::set_stream(const wi_stream &stream){
	this->nfreq = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;
}

void sys_temperature_bandpass::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride){
	ssize_t nf = this->nfreq;
	ssize_t nt = this->nt_chunk;
	// for(int i = 0; i < nt * nf; i++){
	// 	intensity[i] = 0.0f;
	// }
	float* meds = new float[nf];
	#pragma omp parallel for
	for(ssize_t i = 0; i < nf; i++){
		meds[i] = median(&intensity[i*stride],nt);
	}
	
	float medmed = median(meds,nf);
	//float* zero_ar = broadcast(0, nt);
	#pragma omp parallel for
	for(ssize_t i = 0; i < nf; i++){
		if(meds[i] < medmed*0.001){
			memset(&intensity[i*stride],0,nt);
		}
	}
	delete[] meds;

	//delete zero_ar;
	//fin
}

struct remove_bad_times : public wi_transform{
	float sigma_cut;

	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	ssize_t nt_maxwrite;
	remove_bad_times(const float my_sigma_cut);
	void set_stream(const wi_stream &stream);
	void start_substream(int isubstream, double t0){}
	void end_substream(){}
	void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	std::string get_name() const{
		return std::string("remove_bad_times");
	}
};

remove_bad_times::remove_bad_times(const float my_sigma_cut){
	this->name = this->get_name();
	sigma_cut = my_sigma_cut;
}

void remove_bad_times::set_stream(const wi_stream &stream){
	this->nfreq = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;
}

void remove_bad_times::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride){
	ssize_t nt = this->nt_chunk;
	ssize_t nf = this->nfreq;
	float* stds = new float[nt];
	float* means = new float[nf];
	for(ssize_t i = 0; i < nt; i++){
		float* this_t = new float[nt];
		for(ssize_t j = 0; j < nf; j++){
			this_t[j] = intensity[j*stride + i];
		}
		stds[i] = sqrt(sample_var(this_t,nf));
		delete[] this_t;
	}

	for(ssize_t i = 0; i < nf; i++){
		means[i] = mean(&intensity[i*stride],nt);
	}

	float stdstd = sqrt(sample_var(stds,nt));
	float meanstd = mean(stds,nt);
	int reps = 0;
	for(ssize_t i = 0; i < nt; i++){
		if(stds[i] - meanstd > sigma_cut*stdstd){
			reps++;
			for(ssize_t j = 0; j < nf; j++){
				intensity[j*stride + i] = means[j];
			}
		}
	}
	cout << "replaced " << reps << " time channels!" << endl;

	delete[] stds;
	delete[] means;
}

struct png_writer : public wi_transform{
	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	ssize_t nt_maxwrite;
	int chunk = 0;

	char* name_base;
	png_writer(char* name);
	~png_writer(){}
	virtual void set_stream(const wi_stream &stream);
	virtual void start_substream(int isubstream, double t0){}
	virtual void end_substream(){}
	virtual void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	virtual std::string get_name() const{
		return std::string("png_writer");
	}
};

png_writer::png_writer(char* my_name_base){
	this->name = this->get_name();
	name_base = my_name_base;
}

void png_writer::set_stream(const wi_stream &stream){
	this->nfreq = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;
}

static float fmin(float* ar, int len){
	float min = ar[0];
	for(int i = 0; i < len; i++){
		if(ar[i] < min){
			min = ar[i];
		}
	}
	return min;
}

static float fmax(float* ar, int len){
	float max = ar[0];
	for(int i = 0; i < len; i++){
		if(ar[i] > max){
			max = ar[i];
		}
	}
	return max;
}

// void png_writer::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride){
// 	png::image< png::rgb_pixel > image(nfreq, nt_chunk);
// 	int n = nfreq * nt_chunk;
// 	float imax = fmax(intensity,n);
// 	float imin = fmin(intensity,n);
// 	float irange = imax - imin;
// 	int val;
// 	for(uint i = 0; i < nfreq; i++){
// 		for(uint j = 0; j < nt_chunk; j++){
// 			val = (int) (254 * log((intensity[i * nt_chunk + j] - imin)/irange));
// 			image[i][j] = png::rgb_pixel(val, val, val);
// 		}
// 	}
// 	//char name[]
// 	std::string name(name_base);
// 	name += to_string(chunk);
// 	name += std::string(".png");
// 	//strcat(name_base,to_string(chunk).c_str());
// 	//strcat(name_base,".png");
// 	image.write(name.c_str());
// 	chunk++;
// }

double* f_to_d(float* in, int nx, int ny, int stride){
	double* ret = new double[nx * ny];
	for(int i = 0; i < nx ; i++){
		for(int j = 0; j < ny; j++){
			ret[i * ny + j] = (double) in[i * stride + j];
		}
	}
	return ret;
}

//imshow_save_simple(double *ar, int nrow, int ncol, std::string fname);
void png_writer::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride){
	int n = nfreq * nt_chunk;
	double *chunkd = f_to_d(intensity,nfreq,nt_chunk,stride);
	for(int i = 0; i < n; i++){
		chunkd[i] = log(chunkd[i]);
	}
	float imax = fmax(intensity,n);
	float imin = fmin(intensity,n);
	float irange = imax - imin;

	std::string name(name_base);
	name += std::to_string(chunk);
	name += std::string(".png");
	imshow_save_simple(chunkd, nfreq, nt_chunk, name);
	delete[] chunkd;
	chunk++;
}


struct highpass_filter : public wi_transform{
	ssize_t n;
	fftw_complex *window_comp, *window_fft, *nt_input, *nt_fft;
	fftw_plan forward_p;
	fftw_plan reverse_p;

	int window_width;
	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	ssize_t nt_maxwrite;
	//ssize_t nt_chunk;
	highpass_filter(const int width);
	~highpass_filter();
	void set_stream(const wi_stream &stream);
	void start_substream(int isubstream, double t0){}
	void end_substream(){}
	void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	std::string get_name() const{
		return std::string("highpass_filter");
	}
};

highpass_filter::highpass_filter(const int width){
	this->name = this->get_name();
	window_width = static_cast<int>(static_cast<float>(width)/0.4054785f);
	if(window_width % 2 == 1){
		window_width++;
	}
}

highpass_filter::~highpass_filter(){
	delete[] window_comp;
	delete[] window_fft;
	delete[] nt_input;
	delete[] nt_fft;
	//delete[] forward_p;
	//delete[] reverse_p;
}

void highpass_filter::set_stream(const wi_stream &stream){
	this->nfreq = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;
	window_width = min(window_width, (int) this->nt_maxwrite);
	this->nt_prepad = window_width;
	
	n = this->nt_maxwrite + this->nt_prepad;

	window_comp = fftw_alloc_complex(n);
	window_fft = fftw_alloc_complex(n);
	nt_input = fftw_alloc_complex(n);
	nt_fft = fftw_alloc_complex(n);
	fftw_plan window_p = fftw_plan_dft_1d(n, window_comp, window_fft, FFTW_FORWARD, FFTW_ESTIMATE);
	forward_p = fftw_plan_dft_1d(n, nt_input, nt_fft, FFTW_FORWARD, FFTW_MEASURE);
	reverse_p = fftw_plan_dft_1d(n, nt_fft, nt_input, FFTW_BACKWARD, FFTW_MEASURE);

	float* window = new float[n] ();
	for(ssize_t i = 0; i < window_width; i++){
		window[i] = blackman_fun(i,window_width);
	}

	neg_normalize(window,window_width);
	window[window_width/2] += 1;

	//copy window into buffer
	for(ssize_t i = 0; i < n; i++){
		window_comp[i][0] = window[i];
		window_comp[i][1] = 0;
	}

	fftw_execute(window_p);
	//delete[] window_p;
	delete[] window;

	cout << "highpass filter configured with nt_maxwrite " << stream.nt_maxwrite << endl;
	cout << "sample delta t: " << stream.dt_sample << endl;
	// this->nt_prepad = stream.nt_prepad;
	// this->nt_postpad = stream.nt_postpad;
}

void highpass_filter::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride){
	//cout << "someone called me!" << endl;
	//cout << "found " << (t1 - t0) << " seconds of data" << endl;
	//cout << pp_stride << " " << stride << endl;
	ssize_t nt_p = this->nt_prepad;
	for(ssize_t i = 0; i < this->nfreq; i++){
		for(ssize_t j = 0; j < nt_prepad; j++){
			nt_input[j][0] = pp_intensity[i*pp_stride + j];
			nt_input[j][1] = 0;

			nt_fft[j][0] = 0;
			nt_fft[j][1] = 0;
		}
		for(ssize_t j = 0; j < this->nt_chunk; j++){
			nt_input[j + nt_p][0] = intensity[i*stride + j];
			nt_input[j + nt_p][1] = 0;

			nt_fft[j + nt_p][0] = 0;
			nt_fft[j + nt_p][1] = 0;
		}
		fftw_execute(forward_p);
		mult_inplace_complex(nt_fft, window_fft, n);
		fftw_execute(reverse_p);
		for(ssize_t j = 0; j < stride; j++){
			intensity[i*stride + j] = nt_input[j + nt_p][0];
		}
	}

	//cout << "highpass filter finished" << endl;
}

#define DISP_CONST 4.149e3

double disp_delay(double dm, double f)
{
	return DISP_CONST * dm / (f * f);
}

double event_len(double dm, double f0, double f1)
{
	return disp_delay(dm, f1) - disp_delay(dm, f0);
}

//solve for closest freq given time relative to first arrival
double freq_solve(double dm, double t, double f0)
{
	return f0 / std::sqrt(1. + t * (f0 * f0)/(DISP_CONST * dm));
}

class pulse_event
{
	public:
		pulse_event(){};
		double dm;
		double width;
		double t0;
		double t_simulated;
		double t_len;
		double intensity;

	// pulse_event(double mydm, double mywidth, double myt0, double t_simulated, double t_len, double intensity){
	// 	dm = mydm;
	// 	width = mywidth;
	// 	t0 = myt0;
	// 	this->t_simulated = t_simulated;
	// 	this->t_len = t_len;
	// 	this->intensity = intensity;
	// }
};



// struct simple_sim : public wi_transform{
// 	double freq_lo_MHz;
// 	double freq_hi_MHz;
// 	double dt_sample;
// 	double df;
// 	int nsims, nf, nt;
// 	ssize_t nt_maxwrite;
// 	int chunk = 0;

// 	simple_sim(double* dms, double* rates, double* widths, double* intensity, int nsims);
// 	~simple_sim();
// 	void spawn_events(double t0, double t1);
// 	void despawn_events(double t0, double t1);
// 	void add_pulse_subsection(double dm, int width, double intensity, float* ar, int t0, int t1, double pt0, double frame_t0);
// 	double draw_prob();
// 	void set_stream(const wi_stream &stream);
// 	void start_substream(int isubstream, double t0){}
// 	void end_substream(){}
// 	void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
// 	virtual std::string get_name() const{
// 		return std::string("simple_sim");
// 	}
// };

// simple_sim::simple_sim(double* mydms, double* myrates, double* mywidths, double* intensities, int mynsims){
// 	this->name = this->get_name();
// 	this->dms = mydms;
// 	this->rates = myrates;
// 	this->widths = mywidths;
// 	this->intensities = intensities;
// 	this->nsims = mynsims;
// }

struct pulse_sim : public wi_transform{
	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	double df;
	double *dms, *rates, *widths, *intensities;
	int nsims, nf, nt;
	ssize_t nt_maxwrite;
	int chunk = 0;
	std::shared_ptr<std::vector<pulse_event>> active_events;
	//shared_ptr<vector<pulse_event>> active_events(new vector<pulse_event>(0));

	pulse_sim(double* dms, double* rates, double* widths, double* intensity, int nsims);
	~pulse_sim();
	void spawn_events(double t0, double t1);
	void despawn_events(double t0, double t1);
	void add_pulse_subsection(double dm, int width, double intensity, float* ar, int t0, int t1, double pt0, double frame_t0);
	double draw_prob();
	void set_stream(const wi_stream &stream);
	void start_substream(int isubstream, double t0){}
	void end_substream(){}
	void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	virtual std::string get_name() const{
		return std::string("pulse_sim");
	}
};

pulse_sim::pulse_sim(double* mydms, double* myrates, double* mywidths, double* intensities, int mynsims){
	this->name = this->get_name();
	this->dms = mydms;
	this->rates = myrates;
	this->widths = mywidths;
	this->intensities = intensities;
	this->nsims = mynsims;
	active_events = std::shared_ptr<std::vector<pulse_event>>(new std::vector<pulse_event>(0)); 
	std::srand(std::time(0)); // use current time as seed for random generator
}

pulse_sim::~pulse_sim(){
	delete[] dms;
	delete[] rates;
	delete[] widths;
}

void pulse_sim::set_stream(const wi_stream &stream){
	this->nfreq = stream.nfreq;
	this->nf = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;
	this->nt = nt_chunk;
	this->df = (double) ((freq_lo_MHz - freq_hi_MHz)/((double) nf));
}

double pulse_sim::draw_prob(){
	return ((double) std::rand())/((double) RAND_MAX - 1);
}

//pt stands for pulse time, i.e. time relative to
//there is an issue with widths...
void pulse_sim::add_pulse_subsection(double dm, int width, double intensity, float* ar, int t0, int t1, double pt0, double frame_t0){
	
	double fstart, fend;
	int ifstart, ifend;
	double t0_offset = std::max(frame_t0 + t0 * this->dt_sample - pt0, 0.);
	double width_offset = this->dt_sample * width;
	double t_chunk = this->nt_chunk * this->dt_sample;

	//relative to event start
	double t_start = t0_offset;
	double t_end = (t1 - t0) * this->dt_sample + t_start;


	std::cout << "=================\n";
	std::cout << "simulating\n";
	std::cout << "start " << t_start << " end " << t_end << "\n";
	std::cout << "=================\n";

	//don't worry about the labels here
	int lowf0 = (int) ((freq_solve(dm, t_start, this->freq_hi_MHz) - this->freq_hi_MHz)/this->df);
	int lowf1 = (int) ((freq_solve(dm, t_start + width_offset, this->freq_hi_MHz) - this->freq_hi_MHz)/this->df);
	int highf0 = (int) ((freq_solve(dm, t_end, this->freq_hi_MHz) - this->freq_hi_MHz)/this->df);
	int highf1 = (int) ((freq_solve(dm, t_end + width_offset, this->freq_hi_MHz) - this->freq_hi_MHz)/this->df);
	ifstart = std::max(0, std::min(lowf0,lowf1));
	ifend = std::min((int) this->nt_chunk, std::max(highf0,highf1));
	fstart = ifstart * this->df + this->freq_hi_MHz;
	int tindstart;
	int nadd = 0;
	// std::cout << ifstart << " " << ifend << "\n";
	for(int i = ifstart; i < ifend; i++){
		tindstart = t0 + ((int) ((event_len(dm, fstart, this->freq_hi_MHz + i * this->df))/this->dt_sample));
		for(int j = 0; j < width && j + tindstart < this->nt_chunk; j++){
			ar[i * this->nt_chunk + tindstart + j] += intensity;
			nadd += 1;
		}
	}

	// std::cout << "=================\n";
	// std::cout << "added pulse sections " << nadd << "\n";
	// //std::cout << "width " << width << " t_ind " << j + tindstart << "\n";
	// std::cout << "=================\n";
}

void pulse_sim::spawn_events(double t0, double t1){
	double prob, pp;
	for(int i = 0; i < this->nsims; i++){
		prob = rates[i]*(this->dt_sample)*(this->nt_chunk);
		pp = this->draw_prob();


		// std::cout << "=================\n";
		// std::cout << pp << " " << prob << "\n";
		// std::cout << "=================\n";

		if(pp <= prob){

			// std::cout << "=================\n";
			// std::cout << "spawning event\n";
			// std::cout << "=================\n";
			pulse_event this_event;
			this_event.dm = dms[i];
			this_event.t0 = t0 + this->draw_prob() * (t1 - t0);
			this_event.width = widths[i];
			this_event.t_simulated = this_event.t0;
			this_event.intensity = intensities[i];
			this_event.t_len = event_len(dms[i], this->freq_hi_MHz, this->freq_lo_MHz);

			this->active_events->push_back(this_event);
		}
	}
}

void pulse_sim::despawn_events(double t0, double t1){
	std::vector<int> todelete(0);

	int ind = 0;
	for(pulse_event e : *(this->active_events)){
		if(e.t0 + e.t_len < t1) todelete.push_back(ind);
		ind += 1;
	}

	for(int i : todelete){
		// std::cout << "=================\n";
		// std::cout << "despawning event " << i << "\n";
		// std::cout << "=================\n";
		this->active_events->erase(active_events->begin() + i);
	}
}

void pulse_sim::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride){
	this->spawn_events(t0, t1);

	for(auto e : *active_events){
		double tstart, tend;
		int width, tind0, tind1;
		if(t1 > e.t_simulated && e.t_simulated - e.t0 < e.t_len){
			if(e.t_simulated < t0) tstart = t0;
			else tstart = e.t_simulated;

			tend = t1;
			if(e.t0 + e.t_len > tend) tend = e.t0 + e.t_len;

			width = (int) (e.width/this->dt_sample);
			tind0 = (int) ((tstart - t0)/this->dt_sample);
			tind1 = (int) ((tend - t0)/this->dt_sample);
			//double dm, int width, double intensity, float* ar, int t0, int t1, double pt0, double frame_t0
			this->add_pulse_subsection(e.dm, width, e.intensity, intensity, tind0, tind1, e.t0, t0);
			e.t_simulated += tend;
		}
	}

	this->despawn_events(t0, t1);
}

struct noise_inject : public wi_transform{
	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	double df;
	double stdev;
	ssize_t nt_maxwrite;
	std::mt19937 *generator;
	std::normal_distribution<> *d;

	noise_inject(double stdev);
	~noise_inject(){}
	float get_chi2();
	void set_stream(const wi_stream &stream);
	void start_substream(int isubstream, double t0){}
	void end_substream(){}
	void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	virtual std::string get_name() const{
		return std::string("noise_inject");
	}
};

noise_inject::noise_inject(double stdev)
{
	this->name = this->get_name();
	this->stdev = stdev;    
	std::random_device rd;
    	generator = new std::mt19937(rd());
    	d = new std::normal_distribution<>(0.0,stdev);
}

float noise_inject::get_chi2(){
	float tmp, a, b;
	a = (*d)(*generator);
	b = (*d)(*generator);
	return (float) (a*a + b*b);
}

void noise_inject::set_stream(const wi_stream &stream)
{
	this->nfreq = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;
}

void noise_inject::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride)
{
	for(int i = 0; i < this->nt_chunk * this->nfreq; i++){
		intensity[i] += get_chi2();
	}
}


struct spect_integrate : public wi_transform{
	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	double df;
	double stdev;
	ssize_t nt_maxwrite;
	double t_int;
	std::shared_ptr<std::vector<double>> accum;
	int n_int;
	int this_int;
	int chunk = 0;
	std::string name_base;

	spect_integrate(double t_int, std::string basename);
	~spect_integrate(){}
	void set_stream(const wi_stream &stream);
	void start_substream(int isubstream, double t0){}
	void end_substream(){}
	void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	void save_plot();
	virtual std::string get_name() const{
		return std::string("spect_integrate");
	}
};

spect_integrate::spect_integrate(double t_int, std::string name_base)
{
	this->name = this->get_name();
	this->t_int = t_int;
	this->name_base = name_base;
}

void spect_integrate::set_stream(const wi_stream &stream)
{
	this->nfreq = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;

	this->n_int = (int) (this->t_int / (this->dt_sample * this->nt_chunk) + 0.5);
	this->accum = std::shared_ptr<std::vector<double>>(new std::vector<double>(this->nfreq));
}

//void imshow_save_simple(double *ar, int nrow, int ncol, std::string fname);
void spect_integrate::save_plot(){
	std::string fname("");
	fname += name_base;
	fname += to_string(chunk);
	fname += std::string(".png");
	plot_save_simple(&((*accum)[0]), nfreq, fname);
}

void spect_integrate::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride)
{
	for(int i = 0; i < nfreq; i++){
		double sum = 0.0;
		for(int j = 0; j < nt_chunk; j++){
			sum += (double) intensity[i * stride + j];
		}
		(*accum)[i] += sum;
	}

	this_int += 1;

	if(this_int >= n_int){
		this_int = 0;
		this->save_plot();
		accum = std::shared_ptr<std::vector<double>>(new std::vector<double>(this->nfreq));
	}

	chunk++;
}


struct time_intensity : public wi_transform{
	double freq_lo_MHz;
	double freq_hi_MHz;
	double dt_sample;
	double df;
	double stdev;
	ssize_t nt_maxwrite;
	std::shared_ptr<std::vector<double>> accum;
	int chunk = 0;
	std::string name_base;

	time_intensity(std::string basename);
	~time_intensity(){}
	void set_stream(const wi_stream &stream);
	void start_substream(int isubstream, double t0){}
	void end_substream(){}
	void process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride);
	void save_plot();
	virtual std::string get_name() const{
		return std::string("time_intensity");
	}
};

time_intensity::time_intensity(std::string name_base)
{
	this->name = this->get_name();
	this->name_base = name_base;
}

void time_intensity::set_stream(const wi_stream &stream)
{
	this->nfreq = stream.nfreq;
	this->freq_lo_MHz = stream.freq_lo_MHz;
	this->freq_hi_MHz = stream.freq_hi_MHz;
	this->dt_sample = stream.dt_sample;
	this->nt_maxwrite = stream.nt_maxwrite;
	this->nt_chunk = stream.nt_maxwrite;

	//this->n_int = (int) (this->t_int / (this->dt_sample * this->nt_chunk) + 0.5);
	this->accum = std::shared_ptr<std::vector<double>>(new std::vector<double>(this->nt_chunk));
}

//void imshow_save_simple(double *ar, int nrow, int ncol, std::string fname);
void time_intensity::save_plot(){
	std::string fname("");
	fname += name_base;
	fname += to_string(chunk);
	fname += std::string(".png");
	plot_save_simple(&((*accum)[0]), nt_chunk, fname);
}

void time_intensity::process_chunk(double t0, double t1, float* intensity, float* weights, ssize_t stride, float* pp_intensity, float* pp_weight, ssize_t pp_stride)
{
	for(int i = 0; i < nt_chunk; i++){
		double sum = 0.0;
		for(int j = 0; j < nfreq; j++){
			sum += (double) intensity[j * stride + i];
		}
		(*accum)[i] += sum;
	}

	this->save_plot();

	//what happens to the memory?
	accum = std::shared_ptr<std::vector<double>>(new std::vector<double>(this->nfreq));

	// this_int += 1;

	// if(this_int >= n_int){
	// 	this_int = 0;
	// 	this->save_plot();
	// 	accum = std::shared_ptr<std::vector<double>>(new std::vector<double>(this->nfreq));
	// }

	chunk++;
}