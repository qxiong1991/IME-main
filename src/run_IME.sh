#IME-main
python3 IME-main.py \
--IME \
--satellite_UAV_area_ratio_min=-10000 \
--satellite_UAV_area_ratio_max=10000 \
--satellite_UAV_HuMoments2_ratio_min=-2 \
--satellite_UAV_HuMoments2_ratio_max=2 \
--satellite_UAV_HuMoments_ratio_min=-2 \
--satellite_UAV_HuMoments_ratio_max=2 \
--ProjectedModel='Center_projection' \
--output_file='../results/calculated_coordinates_method1.csv' \

python3 IME-main.py \
--IME \
--satellite_UAV_area_ratio_min=-0.9 \
--satellite_UAV_area_ratio_max=2 \
--satellite_UAV_HuMoments2_ratio_min=-2 \
--satellite_UAV_HuMoments2_ratio_max=2 \
--satellite_UAV_HuMoments_ratio_min=-2 \
--satellite_UAV_HuMoments_ratio_max=2 \
--ProjectedModel='Center_projection' \
--output_file='../results/calculated_coordinates_method2.csv' \

python3 plot_data1.py \
