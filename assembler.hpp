#include <vector>
#include <memory>

#include <thread>

#include <queue>
#include <mutex>
#include <condition_variable>

#ifndef RF_HEADER
#define RF_HEADER
#include "rf_pipelines.hpp"
#include "rf_pipelines_internals.hpp"
#endif

using namespace rf_pipelines;

namespace aconst
{
	const int header_size = 32;
	const int payload_count = 1024;
	const int nframes_chunk = 512;
	const int payload_size = 1024 * 4;
	const int packet_size = header_size + payload_size;
	const int nchunk_buf = 16;
};

//typedef std::vector<std::thread<std::shared_ptr<assembler>>> thread_vect;
typedef float buf_dt;

// struct header
// {
// 	//time ind, in 10s units
// 	unsigned long tensec;

// 	//number of frames per 10 seconds
// 	int nframes;

// 	//frame number within 10-second window
// 	int frame_id;

// 	//number of integrations per frame (nint = nframes_native/nframes)
// 	int nint;

// 	//number of valid native frames integrated in chunk
// 	int valid_ints;

// 	//polarization label, arbitrary
// 	bool pol_id;
// };


typedef struct stream_def{
	int packet_length;		// - packet length
	int header_length;		// - header length
	int samples_per_packet;	// - number of samples in packet
	int sample_type;		// - data type of samples in packet
	double raw_cadence;		// - raw sample cadence
	int num_freqs;			// - number of frequencies
	int num_elems;			// - number of elements
	int samples_summed;		// - samples summed for each datum
	uint handshake_idx;		// - frame idx at handshake
	double handshake_utc;	// - UTC time at handshake
	float freq_lo; //MHz
	float freq_hi; //MHz
} stream_def;

class assembler : public wi_stream
{
	public:
		assembler(int sock_fd, std::vector<std::function<std::shared_ptr<wi_transform>()>> trans_factories);
		~assembler();

		bool is_active();
		bool is_connected();
		int sock_fd;

		std::shared_ptr<std::vector<std::shared_ptr<wi_transform>>> transforms;
		virtual void stream_body(wi_run_state &run_state) override;
		void get_intensity_chunk(float *intensity, ssize_t stride);
		void receive_handshake();
		void populate_chunk(std::shared_ptr<std::vector<float>> buf);
		void receive_chunk();
		stream_def stream_params;

		// };


		// int nfreq;
		// int nt_maxwrite;
		// double freq_lo_MHz;
		// double freq_hi_MHz;
		// double dt_sample;
	private:

		//std::vector<unsigned char> tmp_chunk;
		std::shared_ptr<std::queue<std::vector<float>>> chunks;
		int nrec_chunk;
		int nrec_packet;
		bool active = true;
		//bool buf_empty[aconst::nchunk_buf];
		//buf_dt buf[aconst::payload_count * aconst::nframes_chunk * aconst::nchunk_buf];
		//struct header header_buf[aconst::nchunk_buf];
		mutable std::mutex m_a;

};

class assembler_server
{
	public:
		assembler_server(std::string config_file);
		//assembler_server(&assembler_server as);
		~assembler_server();
		void run();
		bool is_connected();
		bool is_active();
		void set_active(bool new_act);

	private:
		int accept_connection();
		std::string addr;
		int port;
		int sock_fd;
		bool active = true;
		bool initialized_assembler = false;
		mutable std::mutex m_a;
		std::shared_ptr<assembler> assmblr;
		std::shared_ptr<std::thread> active_assembler;
		std::vector<std::function<std::shared_ptr<wi_transform>()>> trans_factories;
};