python3 microbenchmark.py --suite=1-to-m --functional-test
python3 microbenchmark.py --suite=n-to-m --functional-test
python3 benchmark.py --suite=unet --comm-overlap-level 0 --exp-name unet.baseline
python3 benchmark.py --suite=unet --comm-overlap-level 1 --exp-name unet.overlap
python3 benchmark.py --suite=unet --comm-overlap-level 2 --exp-name unet.overlap_with_schedule
python3 benchmark.py --suite=gpt --comm-overlap-level 0 --exp-name gpt.baseline
python3 benchmark.py --suite=gpt --comm-overlap-level 1 --exp-name gpt.overlap
python3 benchmark.py --suite=gpt --comm-overlap-level 2 --exp-name gpt.overlap_with_schedule

