from DeepJetCore.DataCollection import DataCollection
import sys

djdc_path = sys.argv[1]

train_data = DataCollection(djdc_path)

# splits off 10% of the training dataset for validation. Can be used in the same way as train_data
val_data=train_data.split(0.9)

# Set the batch size.
# If the data is ragged in dimension 1 (see convert options),
# then this is the maximum number of elements per batch, which could be distributed differently
# to individual examples. E.g., if the first example has 50 elements, the second 48, and the third 30,
# and the batch size is set to 100, it would return the first two examples (in total 99 elements) in
# the first batch etc. This is helpful to avoid out-of-memory errors during training

train_data.setBatchSize(100)

print("batch size: 100")
# prepare the generator

train_data.invokeGenerator()

# loop over epochs here ...

train_data.generator.shuffleFilelist()
train_data.generator.prepareNextEpoch()

# this number can differ from epoch to epoch for ragged data!
nbatches = train_data.generator.getNBatches()

print("nbatches: {}".format(nbatches))

for b in range(1):

    #should not happen unless files are broken (will give additional errors)
    if train_data.generator.isEmpty():
        raise Exception("ran out of data")

    # this returns a TrainData object.
    data = train_data.generator.getBatch()

    features_list = data.transferFeatureListToNumpy()
    truth_list = data.transferTruthListToNumpy()
    weight_list = data.transferWeightListToNumpy() #optional
    print("features")
    print(features_list[0:3])
    print("truth")
    print(truth_list[0:3])
    print("weights")
    print(weight_list[0:3])
    # do your training

# end epoch loop

