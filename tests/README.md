# Unit test

## Requirement
A machine with at least 4 gpus.

## Run all test cases

1. Start a ray cluster
```
ray start --head
```

2. Run all tests
```
python3 run_all.py
```

## Run specific files

- For debug usage:
```
python3 shard_parallel/test_basic.py
```

- More similar to how CI runs files
```
# Run one file
python3 run_all.py --run-pattern shard_parallel/test_basic.py

# Run a folder
python3 run_all.py --run-pattern shard_parallel
```
