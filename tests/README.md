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

## Run a specific file

- For debug usage:
```
python3 test_auto_sharding_basic.py
```

- More similar to how CI runs a file
```
python3 run_all.py --filter test_auto_sharding_basic.py
```
