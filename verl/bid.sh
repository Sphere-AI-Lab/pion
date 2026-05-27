condor_submit_bid 35 -i -append request_cpus=32 -append request_gpus=8 -append request_memory=800000 -append 'requirements = TARGET.CUDAGlobalMemoryMb > 60000'
