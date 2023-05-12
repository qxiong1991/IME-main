# IME-main
# Image moment extraction module selects aerial photos for UAV high-precision geolocation without GPS

###  Abstract

Since GPS is susceptible to environmental interference or cannot be used in some specific situations, it is important to study the UAV geolocation under GPS-free condition. How to ensure that the aerial photos used for positioning have high accuracy has been a concern and needs to be solved urgently. Usually determines whether aerial photos can be used for localization based on the number of feature points matched between aerial photo and satellite photo and whether the aerial photo center projection is in the specially designated area of satellite photo, but this method has low accuracy. In this paper, we propose to select aerial photos for positioning based on the image moment extraction module, and verify the effectiveness and advancement of this method through experiments. Firstly, it is theoretically concluded that the localization method based on the aerial photo center projection is superior to that based on the aerial photo projection area center; secondly, it is theoretically concluded that the aerial photos selected by the image moment extraction module have good characteristics of translation, rotation, scaling and non-super scaling, and can be applied to the UAV high precision geolocation. Finally, the above theory is verified in the UAV flight data set, and it is concluded that under the condition that the UAV adopts the aerial photo center projection for positioning, compared with the first method of selecting aerial photos for positioning, the image moment extraction module can reduce the average absolute error of UAV positioning from 63.220m to 9.454m and the maximum error of the whole course from 486.7m to 47.194m, and the aerial photos used for positioning account for 70.96% of the total number of aerial photos. This paper is of great significance to the realization of UAV high precision geolocation under GPS-free condition.

### Experiment Dataset

Aerial photos can be found [here](https://utufi.sharepoint.com/:f:/s/msteams_0ed7e9/EsXaX0CKlpxIpOzVnUmn8-sB4yvmsxUohqh1d8nWKD9-BA?e=gPca2s).
The satellite map is stored in  "assets/map"

## Run the code

The IME algorithm was tested on Ubuntu 20.04 with Python 3.9.12 and pytorch1.12.1. Nevertheless, it should work with other versions as well.

   0. Create a new python3 virtual environment

      python3 -m venv env 

      source env/bin/activate

   1. Clone the repo
   
      git clone git@github.com:qxiong1991/IME-main.git
 
   2. Install superglue dependencies:        #env and IME-main must be in one folder
   
      cd IME-main
      git submodule update --init --recursive
      
   3. Install python dependencies
   
      pip3 install -r requirements.txt
      
   4. Run the localization script
   
      cd src

   5. experiment 1
   
      sh run.sh
      
   6. experiment 2
   
      sh run_IME_without_Area.sh
      
   7. experiment 3
   
      sh run_IME.sh

## Build your own  dataset

Before running the code, you need to have a dataset, reference images (map) 

   1. Add your UAV photos to ```assets/query```. 

   2. Add your satellite photos to ```assets/map``` together with a csv file containing geodata for the images (see ```assets/map/map.csv```) 

   3. Run python script to generate csv file containing photo metadata with GPS coordinates

      python3 extract_image_meta_exif.py




