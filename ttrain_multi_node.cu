#ifdef MULTI_GPU
#include <nccl.h>
#endif

typedef struct {
    int process_rank;
    int num_processes;
    int local_device_idx;

    int zero_stage;
    size_t shard_num_parameters;
#ifdef MULTI_GPU
    ncclComm_t nccl_comm;
    cudaStream_t nccl_stream;
    cudaEvent_t compute_nccl_sync;
    float* unified_buffer;
#endif
} MultiGpuConfig;

MultiGpuConfig multi_gpu_config_init() {
#ifdef MULTI_GPU
    MultiGpuConfig result;

}
int main(int argc, char* argv[]) {
    const char* train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* load_filename = "gpt2_124M_bf16.bin";
    const char* lr_scheduler_type = "cosine";
    const char* output_log_dir = NULL;

    multi_gpu_config
}