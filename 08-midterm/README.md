<h1 align = "center">
<b><i>Phone Price Prediction</i></b>
</h1>

## Problem Description

Being a Mobile Developer, I'm passionate about all things mobile. Hence, in this Midterm Project, the task to tackle is
how
to predict the price of a phone based on its features like: the Operating System (Android / iOS), the prestige of its
brand (Apple, Samsung, Xiaomi, OPPO) and other technical specifications (battery, screen size, etc).

## Data

The data used in this project is the [**Mobile Phone Specifications and
Prices**](https://www.kaggle.com/datasets/pratikgarai/mobile-phone-specifications-and-prices) dataset, which can be
found in Kaggle.

Here is a detailed look of the features provided in the dataset, with their descriptions
and unit of measure (if applicable).

**Note:** *Some feature names are different from those of the actual dataset, which are cleaned.*

|     Feature Name	      |                   	Feature Description	                    |
|:----------------------:|:----------------------------------------------------------:|
|          Name          |                     Name of the Phone                      |
|         Brand          |                         Brand Name                         |
|         Model          |                        Phone Model                         |
| Battery Capacity (mAh) |                  Battery Capacity in mAh                   |
|  Screen Size (inches)  |       Screen Size in inches across opposite corners        |
|      Touch Screen      |     Whether the phone is touchscreen supported or not      |
|      Resolution X      | The resolution of the phone along the width of the screen  |
|      Resolution Y      | The resolution of the phone along the height of the screen |
|   RAM Processor (MB)   |               RAM available in phone in MB                 |
| Internal Storage (GB)  |              Internal Storage of phone in GB               |
|      Rear Camera       |     Resolution of rear camera in MP (0 if unavailable)     |
|      Front Camera      |    Resolution of front camera in MP (0 if unavailable)     |
|    Operating System    |            The Operating System (Android / iOS)            |
|         Wi-Fi          |           Whether the phone supports WiFi or not           |
|       Bluetooth        |        Whether the phone supports Bluetooth or not         |
|          GPS           |           Whether the phone supports GPS or not            |
|     Number of SIMs     |                  The total number of SIMs                  |
|           3G           |               Whether the phone is 3G or not               |
|         4G/LTE         |               Whether the phone is 4G or not               |
|         Price          |                   Price in Indian Rupees                   |

# Getting Started

This is a set of instructions on setting up this project locally.
To get a local copy up and running follow these simple example steps.

Prerequisites
This is an example of how to list things you need to use this software

* Python
* Pipenv
* Docker
* Windows Subsystem for Linux (if using Windows)

## Installing Dependencies

You can install the dependencies with pipenv, as they are specified in the `Pipfile` and `Pipfile.lock`, by running
the following commands:

```yaml
pipenv install
pipenv shell
```

## Building the Model

You can run the [`train.py`](/models/train.py) file or
the
full [`model_training.ipynb`](/notebooks/model_training.ipynb)
Jupyter Notebook to perform all the steps
required to train the final model used in this project, which is a Gradient Boosted Trees Regressor (XGBoost).

So, in order to train the model, you can run the following command:

```
python train.py
```

## Serving the Model (Locally)

We can serve our model with BentoML and the [`predict.py`](/predict.py) script by running:

```
bentoml serve predict.py:svc
```

This scripts loads the latest model available locally, which can be used in the browser as BentoML automatically
creates a Swagger UI at http://localhost:3000.
The variables expected by the model to predict the price of a phone can be found in
the [`sample_record.json`](/utils/sample_record.json) file.

## Building the Bento and Containerizing the Model

To containerize the model into a Docker Container with all the required dependencies we use BentoML, which facilitates
this procedure.
With BentoML, in order to containerize the model, it's only necessary to specify a [`bentofile.yaml`](/bentofile.yaml)
file which
specifies the project name, owner and all the required dependencies.
After that, we just have to run the command.

```yaml
bentoml build
``` 

Similar to saving a model, a unique version tag will be automatically generated for the newly created Bento.
The output expected is similar to the one shown below:

```yaml
bentoml build

Building BentoML service "phone_price_predictor:dpijemevl6nlhlg6" from build context "/home/user/gallery/quickstart"
Packing model "iris_clf:zy3dfgxzqkjrlgxi"
Locking PyPI package versions..

██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

Successfully built Bento(tag="phone_price_predictor:dpijemevl6nlhlg6")
```

After creating the Bento, we can finally containerize the model by running
```yaml
bentoml containerize phone_price_predictor:latest
```
which will create a Docker image that we can check by running `docker image ls`.

We can run this image by passing `phone_price_predictor:f35knqlbck3zlhfw` to "docker run".
For example: "docker run -it --rm p 3000:3000 phone_price_predictor:f35knqlbck3zlhfw"

*Note: Naturally, the tag can vary*