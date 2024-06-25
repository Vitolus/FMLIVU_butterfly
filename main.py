#%%
import mlcroissant as mlc
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
#%%
# 1. Point to a local or remote Croissant file
url = "https://huggingface.co/api/datasets/fashion_mnist/croissant"
# 2. Inspect metadata
print(mlc.Dataset(url).metadata.to_json())
#%%
# 3. Use Croissant dataset in your ML workload
builder = tfds.dataset_builders.CroissantBuilder(jsonld=url, file_format='array_record')
builder.download_and_prepare()
#%%
# 4. Split for training/testing
train, test = builder.as_data_source(split=['default[:80%]', 'default[80%:]'])
# 5. Display train and test
# use pandas to display the data
train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)
print(train_df.head())
print(test_df.head())