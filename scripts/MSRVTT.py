from datasets import load_dataset

train_ds = load_dataset("friedrichor/MSR-VTT", "train_9k")
test_ds = load_dataset("friedrichor/MSR-VTT", "test_1k")

print(train_ds)