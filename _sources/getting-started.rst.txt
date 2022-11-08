Getting started
===============

This guide trains an interaction network (IN) on the CERN Open Dataset (http://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q) to identify H->bb jets versus QCD background jets.

Installing
----------

Checkout the prebuilt Docker containers.

* If you use a GPU for training:

.. code-block:: bash

   docker pull jmduarte/hbb_interaction_network:gpu

* If you use a CPU for training:

.. code-block:: bash

   docker pull jmduarte/hbb_interaction_network:cpu

Or rebuild the images from the Dockerfiles accordingly.

* If you use a GPU for training:

.. code-block:: bash

   docker build -f Dockerfile.gpu .

* If you use a CPU for training:

.. code-block:: bash

   docker build -f Dockerfile.cpu .

Run the image. Inside of the Docker container, clone the `hbb_interaction_network` git repository and install it.

.. code-block:: bash

        git clone git@github.com:FAIR4HEP/hbb_interaction_network.git
        cd hbb_interaction_network
        pip install -e .


Convert dataset
----------------
To convert the full training dataset

.. code-block:: bash

   python src/data/make_dataset.py --train

and the testing dataset:

.. code-block:: bash

   python src/data/make_dataset.py --test

Training
--------

To run the nominal training on CPU (or replace device with `cuda` to run on GPU):

.. code-block:: bash

   python src/models/train_model.py --batch-size 1024 --epoch 100 --device cpu

Testing
----------

To test the trained model:

.. code-block:: bash

   python src/models/predict_model.py --batch-size 1024
