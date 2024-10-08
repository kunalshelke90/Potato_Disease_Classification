# Potato_Disease_Classification in Deep Learning using Tensorflow

Farmers who grow potatoes are facing drastic economic losses every year due to the various disease that can happen to a potato plant. There are two common diseases, which are known as Early Blight, Late Blight. Early blight is caused by a fungus and late blight is caused by a specific microorganism. If a farmer can detect the cause of the disease in an early stage and apply appropriate treatment, it can save waste and prevent economic loss tremendously.

## Process
#### :one:   Data Acquisition


A team of annotators who work closely with the farmers to collect the images from the fields and annotate the image either it's a healhy potato leaf or if it has any diseases using domain knowledge. The team collected 2152 potato-leaf images in total.

#### :two:   Data Preparation

- **tf dataset**

- **Resize & Scale**

- **Data augmentation**

<details>
<summary> Data Splitting </summary>

- Create function `get_dataset_partitions_tf()` to split data into **train, validate, test**

- Test prepare function

- Check the size of each dataset
     ```sh
     len(train), len(validate), len(test)
     ```
- Call the function, and cache e the 3 data samples
     ```sh
    train = train.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    validate = validate.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    test = test.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
     ```
</details>

#### :three:    Modeling
- Define neural network architecture

- Build model on training dataset and evaluate on train and validate

- Use optimizer to compile

- Fit model on test dataset on evaluate model based on accuracy

- Plot accuracy and loss function of train and validate datasets from all 50 epochs.

- Make prediction on test dataset and save model

- Ajdust neural network architecture and optimizer, using steps above to generate and save new mdoel

- Deploy the top performing model

## Conclusion
The neurol network model has an accuracy of 99% on test dataset, and it's expected to perform with the equivalent accuracy level on future onseen data.