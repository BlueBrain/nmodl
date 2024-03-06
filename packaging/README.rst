Building Python Wheels
======================

Note: This is only slightly adapted from NEURONs
`scripts <https://github.com/neuronsimulator/nrn/tree/master/packaging/python>`__.

NMODL wheels are built in a manylinux2014 image. Since the generic
docker image is very basic (CentOS 7), a new image, which brings updated
cmake3 (3.12), flex and bison was prepared and made available at
https://hub.docker.com/r/bluebrain/nmodl (tag: wheel).

Setting up Docker
-----------------

`Docker <https://en.wikipedia.org/wiki/Docker_(software)>`__ is required
for building Linux wheels. You can find instructions to setup Docker on
Linux `here <https://docs.docker.com/engine/install/ubuntu/>`__ and on
OS X `here <https://docs.docker.com/docker-for-mac/install/>`__. On
Ubuntu system we typically do:

.. code::sh

   sudo apt install docker.io
   sudo groupadd docker
   sudo usermod -aG docker $USER

Logout and log back in to have docker service properly configured.

Launch the wheel building
-------------------------

For building the wheel we use the ``cibuilwwheel`` utility, which can be installed using:

.. code::sh

   pip install cibuildwheel

Then to build a wheel for the current platform, run:

.. code::sh

   cibuildwheel

If you have Docker installed, you can also build the Linux wheels using:

.. code::sh

   cibuildwheel --platform linux

Note that, if you happen to have Podman installed instead of Docker, you can
set ``CIBW_CONTAINER_ENGINE=podman`` to use Podman instead of Docker for this
task.

Furthermore, in order to build wheels on MacOS, you need to install an official
CPython installer from `python.org <https://www.python.org>`__.

For a complete list of all available customization options for
``cibuildwheel``, please consult the
`documentation <https://cibuildwheel.readthedocs.io/en/stable/options/>`__.

Testing the wheel
-----------------

On MacOS, the testing of the wheel is launched automatically when running
``cibuildwheel``. On Linux, you will need to test the wheel manually, please
see :ref:`testing-installed-module` for the instructions.


Updating the NMODL Docker image
-------------------------------

If you have changed the Dockerfile, you can build the new image as:

.. code::sh

   docker build -t bluebrain/nmodl:tag .

and then push the image to hub.docker.com as:

.. code::sh

   docker login --username=<username>
   docker push bluebrain/nmodl:tag
