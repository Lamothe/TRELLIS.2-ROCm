# TRELLIS.2 on the Strix Halo
A ROCm implementation of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) from Microsoft.

Tested on:
* Fedora 43
* Strix Halo (128 GB)

Note: this repo is a temporary holding place for my work.  It will be integrated to my HOWTO later.

Create and set up the Ubuntu 24.04 container.
```
./start.sh
```
Once inside the container, set up TRELLIS.2.  This will take more than 10 minutes and will require you to supply an HF token.  You can do this by either entering it into the terminal when prompted OR create an `hf.txt` file (git ignored) with the token in it.
```
./setup.sh
```
If everything builds then you can run the TRELLIS.2 example.
```
./example.sh
```
Cleaning up is easy
```
sudo rm -rf build
```