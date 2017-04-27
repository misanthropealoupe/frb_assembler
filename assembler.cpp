#include "assembler.hpp"


#ifndef RF_HEADER
#define RF_HEADER
#include "rf_pipelines.hpp"
#include "rf_pipelines_internals.hpp"
#endif

#include <stdexcept>
#include <utility>

#include <sys/time.h>
#include <chrono>

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>

#include "yaml-cpp/yaml.h"
#include "preprocessing.cpp"


using namespace rf_pipelines;

assembler::assembler(int sock_fd, std::vector<std::function<std::shared_ptr<wi_transform>()>> trans_factories)
	: sock_fd(sock_fd),
	m_a()
{
	this->receive_handshake();

	std::cout << "constructing assembler" << std::endl;

	chunks = std::shared_ptr<std::queue<std::vector<float>>>(new std::queue<std::vector<float>>());
	//get params from config file!!

	transforms = std::shared_ptr<std::vector<std::shared_ptr<wi_transform>>>(new std::vector<std::shared_ptr<wi_transform>>());

	//for(std::function<std::shared_ptr<wi_transform>()> factory : trans_factories){
	for(auto& factory : trans_factories){
		std::cout << "adding transform" << std::endl;
		transforms->push_back(factory());
	}
}

// struct  __attribute__((__packed__)) IntensityHeader {
// 	int packet_length;		// - packet length
// 	int header_length;		// - header length
// 	int samples_per_packet;	// - number of samples in packet
// 	int sample_type;		// - data type of samples in packet
// 	double raw_cadence;		// - raw sample cadence
// 	int num_freqs;			// - number of frequencies
// 	int num_elems;			// - number of elements
// 	int samples_summed;		// - samples summed for each datum
// 	uint handshake_idx;		// - frame idx at handshake
// 	double handshake_utc;	// - UTC time at handshake
// };

// struct  __attribute__((__packed__)) IntensityPacketHeader {
// 	int frame_idx;			//- frame idx
// 	int elem_idx;			//- elem idx
// 	int samples_summed;		//- number of samples integrated
// };

void assembler::receive_handshake()
{
	int nrec_A = 8*sizeof(int) + 2*sizeof(double);
	int rec = 0;
	std::vector<unsigned char> a_buf(nrec_A);
	while(rec < nrec_A){
		rec += read(sock_fd, &(a_buf[rec]), nrec_A - rec);
	}
	memcpy(&stream_params, &(a_buf[0]), nrec_A);

	//nfreq*2*sizeof(float) + nelem*sizeof(char)
	int nrec_B = stream_params.num_freqs*2*sizeof(float) + stream_params.num_elems*sizeof(char);
	std::vector<unsigned char> b_buf(nrec_B);
	rec = 0;
	std::cout << "nfreq " << stream_params.num_freqs << std::endl;
	std::cout << "raw cadence " << stream_params.raw_cadence << std::endl;
	std::cout << "effective cadence " << stream_params.samples_summed * stream_params.raw_cadence << std::endl;
	//don't deal with all this data, for now.
	while(rec < nrec_B){
		rec += read(sock_fd, &(b_buf[rec]), nrec_B - rec);
	}
	//memcpy(&stream_params, &(b_buf[0]), nrec_B);
	memcpy(&(stream_params.freq_lo), &(b_buf[0]), sizeof(float));
	memcpy(&(stream_params.freq_hi), &(b_buf[(stream_params.num_freqs*2 - 1)*sizeof(float)]), sizeof(float));
	stream_params.freq_lo *= 1e-6;
	stream_params.freq_hi *= 1e-6;

	float a, b;
	a = std::min(stream_params.freq_lo, stream_params.freq_hi);
	b = std::max(stream_params.freq_lo, stream_params.freq_hi);
	stream_params.freq_lo = a;
	stream_params.freq_hi = b;

	std::cout << "freq_lo " << stream_params.freq_lo << std::endl;
	std::cout << "freq_hi " << stream_params.freq_hi << std::endl;
	std::cout << "param 0" << (bool) b_buf[nrec_B - 2] << std::endl;
	std::cout << "param 1" << (bool) b_buf[nrec_B - 1] << std::endl;

	nrec_packet = 3*sizeof(int) + stream_params.num_freqs*sizeof(float);
	nrec_chunk = stream_params.num_elems*nrec_packet*aconst::nframes_chunk;

	this->nfreq = stream_params.num_freqs;
	this->freq_lo_MHz = stream_params.freq_lo;
	this->freq_hi_MHz = stream_params.freq_hi;
	this->dt_sample = stream_params.raw_cadence*stream_params.samples_summed;	
	this->nt_maxwrite = aconst::nframes_chunk;
}

void assembler::populate_chunk(std::shared_ptr<std::vector<float>> ret)
{
	std::cout << "attempting to receive chunk data" << std::endl;
	//std::vector<unsigned char> tmp_chunk(nrec_chunk);
	auto tmp_chunk = std::shared_ptr<std::vector<unsigned char>>(new std::vector<unsigned char>(nrec_chunk));
	//std::cout << tmp_chunk.size() << std::endl;
	int rec = 0;
	while(rec < nrec_chunk){
		//std::cout << "rec: " << rec << std::endl;
		rec += read(sock_fd, &((*tmp_chunk)[rec]), nrec_chunk - rec);
	}

	std::cout << "received chunk data" << std::endl;

	// for(int i = 0; i < 10000; i++){
	// 	std::cout << *(&(tmp_chunk[0]) + i);
	// }

	// std::cout << std::endl;
	int nfloats = aconst::nframes_chunk * stream_params.num_freqs * stream_params.num_elems;

	//std::vector<float> tmp_f(aconst::nframes_chunk * stream_params.num_freqs * stream_params.num_elems);
	//float* tmp_fa = reinterpret_cast<float*>(&tmp_chunk[0]);
	//std::memcpy(&(tmp_f[0]), &(tmp_chunk[0]), aconst::nframes_chunk * stream_params.num_freqs * stream_params.num_elems * sizeof(float));
	
	int nfloat_packet = stream_params.num_freqs + 3;
	int nfloat = nrec_chunk/sizeof(float);
	int this_summed;
	//float* tmp_fa = new float[nfloat];
	auto tmp_fa = std::shared_ptr<std::vector<float>>(new std::vector<float>(nfloat));
	float fintegrate = (float) stream_params.samples_summed;
	float this_norm;
	for(int i = 0; i < nfloat; i++){
		if(i % nfloat_packet == 2){
			this_summed = (int) (((*tmp_chunk)[4*i + 3] << 24) | ((*tmp_chunk)[4*i + 2] << 16) | ((*tmp_chunk)[4*i + 1] << 8) | (*tmp_chunk)[4*i]);
			//std::cout << this_summed << std::endl;
			this_norm = fintegrate/((float) this_summed);
		}
		(*tmp_fa)[i] = (float) (((*tmp_chunk)[4*i + 3] << 24) | ((*tmp_chunk)[4*i + 2] << 16) | ((*tmp_chunk)[4*i + 1] << 8) | (*tmp_chunk)[4*i]);
		(*tmp_fa)[i] *= this_norm;
		//std::cout << tmp_fa[i] << std::endl;
	}

	// for(int i = 0; i < 1000; i++){
	// 	std::cout << tmp_fa[i] << " ";
	// }
	// std::cout << std::endl;
	//memcpy(&tmp_fa[0], &tmp_chunk[0], nrec_chunk);
	// for(int i = 0; i < 1000; i++){
	// 	std::cout << tmp_fa[i] << " ";
	// }
	// std::cout << std::endl;
	// for(int i = 0; i < 1000; i++){
	// 	std::cout << tmp_chunk[i] << " ";
	// }

	//float val = (float) ((0x000000FF & tmp_chunk[16]) | (0x0000FF00 & tmp_chunk[17]) | (0x00FF0000 & tmp_chunk[18]) | (0xFF000000 & tmp_chunk[19]));
	//std::cout << val << std::endl;
	//std::copy(tmp_chunk.begin(), tmp_chunk.begin() + nrec_chunk,
	//std::vector<float> ret(aconst::nframes_chunk * stream_params.num_freqs);
	//std::vector<float> tmp(aconst::nframes_chunk * stream_params.num_freqs);

	//DOOO
	// for(int i = 0; i < stream_params.num_elems; i++){
	// 	memcpy(&(tmp[0]), &(tmp_chunk[3*sizeof(int) + i * stream_params.num_freqs * sizeof(float)]), stream_params.num_freqs * sizeof(float));
	// 	for(int j = 0; j < stream_params.num_freqs; j++){
	// 		ret[j] += tmp[j];
	// 	}
	// }
	//float* tmp = new float[1];
	int index = 0;
	//Lazy loop does not check for ordering of packets and elements
	//Last element of every packet sucks?
	for(int i = 0; i < stream_params.num_freqs - 1; i++){
		for(int j = 0; j < aconst::nframes_chunk; j++){
			//std::cout << "n_integrate: " << (int) tmp_fa[ j * nfloat_packet * stream_params.num_elems + 2] << endl;
			for(int k = 0; k < stream_params.num_elems; k++){
				//memcpy(&(tmp[i * aconst::nframes_chunk]), &(tmp_chunk[3*sizeof(int) + ]), sizeof(float));

				//memcpy(&(tmp[0]), &(tmp_chunk[j * stream_params.num_elems*nrec_packet + k*nrec_packet + i*sizeof(float) + 3*sizeof(int)]), sizeof(float));
				//ret[i*aconst::nframes_chunk + j] += tmp[0];
				//index = j*stream_params.num_elems*nrec_packet + k*nrec_packet + i*sizeof(float) + 3*sizeof(int);
				//index = j * stream_params.num_freqs * (stream_params.num_elems + 3) + k * (stream_params.num_freqs + 3) + i + 3;
				
				index = j * nfloat_packet * stream_params.num_elems + k * nfloat_packet + i + 3;

				//std::cout << tmp_fa[index] << std::endl;
				(*ret)[i*aconst::nframes_chunk + j] += (*tmp_fa)[index];
				//std::cout << index << " " << nrec_chunk << std::endl;
				//std::cout << "val " << ((float*) &(tmp_chunk[index]))[0] << std::endl;
				//ret[i*aconst::nframes_chunk + j] += ((float*) &(tmp_chunk[index]))[0];
			}
		}
	}

	//delete[] tmp_fa;
	//chunks->push(ret);
}

void assembler::receive_chunk()
{
	std::cout << "attempting to receive chunk data" << std::endl;
	std::vector<unsigned char> tmp_chunk(nrec_chunk);
	//std::cout << tmp_chunk.size() << std::endl;
	int rec = 0;
	while(rec < nrec_chunk){
		//std::cout << "rec: " << rec << std::endl;
		rec += read(sock_fd, &(tmp_chunk[rec]), nrec_chunk - rec);
	}

	std::cout << "received chunk data" << std::endl;

	// for(int i = 0; i < 10000; i++){
	// 	std::cout << *(&(tmp_chunk[0]) + i);
	// }

	// std::cout << std::endl;
	int nfloats = aconst::nframes_chunk * stream_params.num_freqs * stream_params.num_elems;

	//std::vector<float> tmp_f(aconst::nframes_chunk * stream_params.num_freqs * stream_params.num_elems);
	//float* tmp_fa = reinterpret_cast<float*>(&tmp_chunk[0]);
	//std::memcpy(&(tmp_f[0]), &(tmp_chunk[0]), aconst::nframes_chunk * stream_params.num_freqs * stream_params.num_elems * sizeof(float));
	
	int nfloat_packet = stream_params.num_freqs + 3;
	int nfloat = nrec_chunk/sizeof(float);
	int this_summed;
	//float* tmp_fa = new float[nfloat];
	std::vector<float> tmp_fa(nfloat);
	float fintegrate = (float) stream_params.samples_summed;
	float this_norm;
	for(int i = 0; i < nfloat; i++){
		if(i % nfloat_packet == 2){
			this_summed = (int) ((tmp_chunk[4*i + 3] << 24) | (tmp_chunk[4*i + 2] << 16) | (tmp_chunk[4*i + 1] << 8) | tmp_chunk[4*i]);
			//std::cout << this_summed << std::endl;
			this_norm = fintegrate/((float) this_summed);
		}
		tmp_fa[i] = (float) ((tmp_chunk[4*i + 3] << 24) | (tmp_chunk[4*i + 2] << 16) | (tmp_chunk[4*i + 1] << 8) | tmp_chunk[4*i]);
		tmp_fa[i] *= this_norm;
		//std::cout << tmp_fa[i] << std::endl;
	}
	// for(int i = 0; i < 1000; i++){
	// 	std::cout << tmp_fa[i] << " ";
	// }
	// std::cout << std::endl;
	//memcpy(&tmp_fa[0], &tmp_chunk[0], nrec_chunk);
	// for(int i = 0; i < 1000; i++){
	// 	std::cout << tmp_fa[i] << " ";
	// }
	// std::cout << std::endl;
	// for(int i = 0; i < 1000; i++){
	// 	std::cout << tmp_chunk[i] << " ";
	// }

	//float val = (float) ((0x000000FF & tmp_chunk[16]) | (0x0000FF00 & tmp_chunk[17]) | (0x00FF0000 & tmp_chunk[18]) | (0xFF000000 & tmp_chunk[19]));
	//std::cout << val << std::endl;
	//std::copy(tmp_chunk.begin(), tmp_chunk.begin() + nrec_chunk,
	std::vector<float> ret(aconst::nframes_chunk * stream_params.num_freqs);
	//std::vector<float> tmp(aconst::nframes_chunk * stream_params.num_freqs);

	//DOOO
	// for(int i = 0; i < stream_params.num_elems; i++){
	// 	memcpy(&(tmp[0]), &(tmp_chunk[3*sizeof(int) + i * stream_params.num_freqs * sizeof(float)]), stream_params.num_freqs * sizeof(float));
	// 	for(int j = 0; j < stream_params.num_freqs; j++){
	// 		ret[j] += tmp[j];
	// 	}
	// }
	//float* tmp = new float[1];
	int index = 0;
	//Lazy loop does not check for ordering of packets and elements
	//Last element of every packet sucks?
	for(int i = 0; i < stream_params.num_freqs - 1; i++){
		for(int j = 0; j < aconst::nframes_chunk; j++){
			//std::cout << "n_integrate: " << (int) tmp_fa[ j * nfloat_packet * stream_params.num_elems + 2] << endl;
			for(int k = 0; k < stream_params.num_elems; k++){
				//memcpy(&(tmp[i * aconst::nframes_chunk]), &(tmp_chunk[3*sizeof(int) + ]), sizeof(float));

				//memcpy(&(tmp[0]), &(tmp_chunk[j * stream_params.num_elems*nrec_packet + k*nrec_packet + i*sizeof(float) + 3*sizeof(int)]), sizeof(float));
				//ret[i*aconst::nframes_chunk + j] += tmp[0];
				//index = j*stream_params.num_elems*nrec_packet + k*nrec_packet + i*sizeof(float) + 3*sizeof(int);
				//index = j * stream_params.num_freqs * (stream_params.num_elems + 3) + k * (stream_params.num_freqs + 3) + i + 3;
				
				index = j * nfloat_packet * stream_params.num_elems + k * nfloat_packet + i + 3;

				//std::cout << tmp_fa[index] << std::endl;
				ret[i*aconst::nframes_chunk + j] += tmp_fa[index];
				//std::cout << index << " " << nrec_chunk << std::endl;
				//std::cout << "val " << ((float*) &(tmp_chunk[index]))[0] << std::endl;
				//ret[i*aconst::nframes_chunk + j] += ((float*) &(tmp_chunk[index]))[0];
			}
		}
	}
	//delete[] tmp_fa;
	chunks->push(ret);
}

void assembler::get_intensity_chunk(float *intensity, ssize_t stride)
{
	std::vector<float> chunk_data(chunks->front());
	chunks->pop();

	for(int i = 0; i < stream_params.num_freqs; i++){
		for(int j = 0; j < aconst::nframes_chunk; j++){
			intensity[i * stride + j] = chunk_data[i * aconst::nframes_chunk + j];
		}
	}
}

void assembler::stream_body(wi_run_state &run_state)
{
	//tmp_chunk = *(new std::vector<unsigned char>(nrec_chunk));
	//receive_chunk();
	//std::thread rec_thr(&assembler::receive_chunk, this);
	//rec_thr.detach();
	//TODO set time properly
	auto tmp_buf = std::shared_ptr<std::vector<float>>(new std::vector<float>(aconst::nframes_chunk * stream_params.num_freqs));
	run_state.start_substream(0.0);
	//std::thread main_t(&assembler::run, this);
			
	for (;;) {	
		float *intensity;
		float *weights;
		ssize_t stride;
		bool zero_flag = false;
		
		run_state.setup_write(nt_maxwrite, intensity, weights, stride, zero_flag);

		tmp_buf->clear();
		populate_chunk(tmp_buf);
		for(int i = 0; i < stream_params.num_freqs; i++){
			for(int j = 0; j < aconst::nframes_chunk; j++){
				intensity[i * stride + j] = (*tmp_buf)[i * aconst::nframes_chunk + j];
			}
		}

		// get_intensity_chunk(intensity,stride);
		
		//initialize weight to 1.0
		for (int i = 0; i < nfreq; i++) {
			for (int j = 0; j < nt_maxwrite; j++) {		
				weights[i*stride + j] = 1.0;
			}
		}
		// std::this_thread::sleep_for(std::chrono::milliseconds(500));
		std::cout << "Bonsai received a chunk." << std::endl;

		run_state.finalize_write(nt_maxwrite);
	}
	run_state.end_substream();
	//main_t.join();
}

bool assembler::is_active(){
	std::lock_guard<std::mutex> lock(m_a);
	return active;
}

bool assembler::is_connected(){
	int error_code;
	int error_code_size = sizeof(error_code);

	//is this true?
	return getsockopt(sock_fd, SOL_SOCKET, SO_ERROR, &error_code, &error_code_size) == 0;
}

assembler::~assembler(){
	close(sock_fd);
	std::lock_guard<std::mutex> lock(m_a);
	active = false;
}

std::function<std::shared_ptr<wi_transform>()> get_transform_factory(std::string name, std::vector<YAML::Node> params)
{
	std::function<std::shared_ptr<wi_transform>()> fun;
	if(name.compare(std::string("png_writer")) == 0){
		fun = [=]() {
			std::shared_ptr<wi_transform> ptr(new png_writer( (params[0].as<std::string>()).c_str() ));
			return ptr;
		};
	}
	else if(name.compare(std::string("bonsai")) == 0){
		fun = [=]() {
			return make_bonsai_dedisperser(params[0].as<std::string>());
		};
	}
	else{
		throw std::runtime_error(std::string("unrecognized transform name: ") + name);
	}

	return fun;
}

assembler_server::assembler_server(std::string config_file)
 : m_a()
{
	//TODO get params from config file
	//TODO come up with wi_transform factory list (lambdas [=,])
	YAML::Node config = YAML::LoadFile(config_file.c_str());
	std::cout << config << std::endl;

	addr = config["address"].as<std::string>();
	port = config["port"].as<int>();
	std::cout << "server address: " << addr << std::endl;
	std::cout << "server port: " << port << std::endl;
	YAML::Node trans_seq = config["transforms"];

	for(auto filter : trans_seq){
		//std::cout << ((YAML::Node) filter) << std::endl;
  		std::string trans_name = filter[0].as<std::string>();
  		std::vector<YAML::Node> params;
  		int ielem = 0;
		for(auto param : (YAML::Node) filter){
			if(ielem > 0) params.push_back((YAML::Node) param);
			ielem++;
		}
		trans_factories.push_back(get_transform_factory(trans_name, params));
	}

	//std::cout << "factory size: " << trans_factories.size() << std::endl;

	struct sockaddr_in serv_addr;
	int portno;

	sock_fd = socket(AF_INET, SOCK_STREAM, 0);
	if (sock_fd < 0){
		throw std::runtime_error("unable to make socket");
	}

	bzero((char *) &serv_addr, sizeof(serv_addr));
	//portno = atoi(port);
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = INADDR_ANY;
	serv_addr.sin_port = htons(port);
	if (bind(sock_fd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
		throw std::runtime_error("unable to bind address");
}

assembler_server::~assembler_server()
{
	// if(initialized_assembler){
	// 	close(active_assembler->sock_fd);
	// }
	close(sock_fd);
}

void assembler_server::set_active(bool new_act)
{
	std::lock_guard<std::mutex> lock(m_a);
	active = new_act;
}

bool assembler_server::is_active()
{
	std::lock_guard<std::mutex> lock(m_a);
	return active;
}

bool assembler_server::is_connected()
{
	if(initialized_assembler == false) return false;
	return assmblr->is_connected();
}

int assembler_server::accept_connection()
{
     int newsockfd;
     socklen_t clilen;
     struct sockaddr_in cli_addr;
     int n;

     //backlog of nominal value 1. should never exceed 1
     listen(sock_fd, 1);

     clilen = sizeof(cli_addr);

     newsockfd = accept(sock_fd, 
                 (struct sockaddr *) &cli_addr, 
                 &clilen);

    return newsockfd;
}

//millis
#define LOOP_SLEEP 100

void assembler_server::run()
{
	while(is_active()){
		std::cout << "assembler server active" << std::endl;
		if(is_connected()){
			std::this_thread::sleep_for(std::chrono::milliseconds(LOOP_SLEEP));
		}
		else{
			std::cout << "attempting a new connection" << std::endl;
			//get new connected socket
			//blocking
			int newsock_fd = accept_connection();

			std::cout << "got connection" << std::endl;
			//spin off new thread
			assmblr = std::shared_ptr<assembler>(new assembler(newsock_fd, trans_factories));
			
			std::cout << "nfreq " << assmblr->nfreq << std::endl;

			//std::shared_ptr<wi_stream> assmblr1 = std::make_shared<assembler>(newsock_fd, std::string("nofile.dat"));

			//std::shared_ptr<wi_stream> assmblr1 = assmblr;

			//std::vector<std::shared_ptr<wi_transform>> trans;

			//assmblr->run(trans);
			//((wi_stream) *assmblr).run(trans);

			std::cout << "starting stream" << std::endl;
			assmblr->run(*(assmblr->transforms));

			//TODO add transforms from factory list

			//TODO start assmblr stream (it creates its own thread)

			//does this increment reference count?
			//active_assembler = std::shared_ptr<std::thread>(new std::thread(((wi_stream) *assmblr).run));
		}
	}
}

int main(int argc, char** argv)
{
	assembler_server a_s(std::string("config.yaml"));

	std::thread thr(&assembler_server::run, &a_s);

	thr.join();
}